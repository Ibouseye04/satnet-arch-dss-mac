#!/usr/bin/env python3
"""
Training Script for SatelliteEvolveGCN Model.

This script trains the EvolveGCN-O model on the SatNetTemporalDataset
for binary classification of satellite network partition risk.

Usage:
    python scripts/train_gnn_model.py [--epochs 20] [--lr 0.01] [--data-dir data/]

Output:
    - Trained model saved to models/satellite_evolvegcn.pt
    - Training metrics logged to console
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

# Add src to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satnet.models.gnn_dataset import SatNetTemporalDataset
from satnet.models.gnn_model import SatelliteEvolveGCN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SatelliteEvolveGCN on temporal satellite network graphs."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Directory containing tier1_design_runs.csv (default: data/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate for Adam optimizer (default: 0.01)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension for EvolveGCN (default: 64)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/satellite_evolvegcn.pt",
        help="Path to save trained model (default: models/satellite_evolvegcn.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cpu', 'cuda', 'mps', or 'auto' (default: auto)",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Determine the best available device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


def move_data_to_device(data_sequence: List[Data], device: torch.device) -> List[Data]:
    """Move a sequence of Data objects to the specified device."""
    return [data.to(device) for data in data_sequence]


def train_epoch(
    model: SatelliteEvolveGCN,
    train_indices: List[int],
    dataset: SatNetTemporalDataset,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: The SatelliteEvolveGCN model
        train_indices: Indices of training samples
        dataset: The SatNetTemporalDataset
        optimizer: Adam optimizer
        criterion: CrossEntropyLoss
        device: Torch device
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for idx in train_indices:
        # Get temporal sequence for this run
        data_sequence = dataset[idx]
        data_sequence = move_data_to_device(data_sequence, device)
        
        # Get label from the first Data object (same for all time steps)
        label = data_sequence[0].y.to(device)  # [1]
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(data_sequence)  # [1, 2]
        
        # Compute loss
        loss = criterion(logits, label)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += 1
    
    avg_loss = total_loss / len(train_indices)
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: SatelliteEvolveGCN,
    test_indices: List[int],
    dataset: SatNetTemporalDataset,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, dict]:
    """
    Evaluate the model on test data.
    
    Args:
        model: The SatelliteEvolveGCN model
        test_indices: Indices of test samples
        dataset: The SatNetTemporalDataset
        criterion: CrossEntropyLoss
        device: Torch device
    
    Returns:
        Tuple of (average_loss, accuracy, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # For confusion matrix
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for idx in test_indices:
        data_sequence = dataset[idx]
        data_sequence = move_data_to_device(data_sequence, device)
        label = data_sequence[0].y.to(device)
        
        logits = model(data_sequence)
        loss = criterion(logits, label)
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += 1
        
        # Update confusion matrix
        pred_val = pred.item()
        label_val = label.item()
        if pred_val == 1 and label_val == 1:
            true_positives += 1
        elif pred_val == 1 and label_val == 0:
            false_positives += 1
        elif pred_val == 0 and label_val == 0:
            true_negatives += 1
        else:
            false_negatives += 1
    
    avg_loss = total_loss / len(test_indices)
    accuracy = correct / total
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }
    
    return avg_loss, accuracy, metrics


def main():
    """Main training loop."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Resolve paths
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    
    output_path = Path(args.output_model)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    try:
        dataset = SatNetTemporalDataset(root=str(data_dir))
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error("Please ensure tier1_design_runs.csv exists in the data directory.")
        sys.exit(1)
    
    num_samples = len(dataset)
    neg_count, pos_count = dataset.get_label_distribution()
    logger.info(f"Dataset: {num_samples} samples (Robust: {neg_count}, Partitioned: {pos_count})")
    
    # Split train/test: simple 80/20 split (first 1600 train, last 400 test)
    split_idx = int(num_samples * 0.8)
    train_indices = list(range(split_idx))
    test_indices = list(range(split_idx, num_samples))
    logger.info(f"Train/Test split: {len(train_indices)}/{len(test_indices)}")
    
    # Initialize model
    model = SatelliteEvolveGCN(
        node_features=3,
        hidden_channels=args.hidden_dim,
        out_channels=2,
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info("-" * 60)
    
    best_test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_indices, dataset, optimizer, criterion, device
        )
        
        # Evaluate
        test_loss, test_acc, metrics = evaluate(
            model, test_indices, dataset, criterion, device
        )
        
        # Log progress
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {metrics['f1']:.4f}"
        )
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "metrics": metrics,
                "args": vars(args),
            }, output_path)
    
    logger.info("-" * 60)
    logger.info(f"Training complete!")
    logger.info(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_epoch}")
    logger.info(f"Model saved to: {output_path}")
    
    # Final evaluation with detailed metrics
    logger.info("\nFinal Test Metrics:")
    test_loss, test_acc, metrics = evaluate(
        model, test_indices, dataset, criterion, device
    )
    logger.info(f"  Accuracy:  {test_acc:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  Confusion Matrix:")
    logger.info(f"    TP: {metrics['true_positives']:4d}  FP: {metrics['false_positives']:4d}")
    logger.info(f"    FN: {metrics['false_negatives']:4d}  TN: {metrics['true_negatives']:4d}")


if __name__ == "__main__":
    main()
