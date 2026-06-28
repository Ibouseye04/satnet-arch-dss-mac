#!/usr/bin/env python3
"""
Training Script for SatelliteGNN Model.

This script trains the GCLSTM model on the SatNetTemporalDataset
for supported resilience targets (classification and regression).

Usage:
    python scripts/train_gnn_model.py [--epochs 20] [--lr 0.01] [--data-dir data/]

Output:
    - Trained model saved to models/satellite_gnn.pt
    - Structured experiment log entry
    - Prediction CSV with stable identifiers
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data

# Add src to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_MODEL = "models/satellite_gnn.pt"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satnet.models.gnn_dataset import SatNetTemporalDataset
try:
    from satnet.models.gnn_model import SatelliteGNN
    GNN_MODEL_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:
    SatelliteGNN = None  # type: ignore[assignment]
    GNN_MODEL_IMPORT_ERROR = exc
from satnet.metrics.resilience_targets import ALL_TARGETS, infer_task_type
from satnet.utils.experiment_logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _validated_config_hash(config: dict[str, Any], *, row_idx: int) -> str:
    if "config_hash" not in config:
        raise ValueError(
            "Prediction export requires `config_hash` for stable ranking joins. "
            f"Dataset row {row_idx} is missing `config_hash`."
        )
    value = config["config_hash"]
    if pd.isna(value):
        raise ValueError(
            "Prediction export requires non-null, non-empty `config_hash` values "
            f"for stable ranking joins. Dataset row {row_idx} has an invalid "
            "`config_hash`."
        )
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(
            "Prediction export requires non-null, non-empty `config_hash` values "
            f"for stable ranking joins. Dataset row {row_idx} has an invalid "
            "`config_hash`."
        )
    return normalized


def _collect_prediction_join_keys(dataset: SatNetTemporalDataset) -> dict[int, str]:
    return {
        idx: _validated_config_hash(dataset.get_run_config(idx), row_idx=idx)
        for idx in range(len(dataset))
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SatelliteGNN on temporal satellite network graphs."
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
        help="Hidden dimension for GCLSTM (default: 64)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation/model selection (default: 0.1)",
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
        default=DEFAULT_OUTPUT_MODEL,
        help="Path to save trained model (default: models/satellite_gnn.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cpu', 'cuda', 'mps', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Load cached graph sequences when available",
    )
    parser.add_argument(
        "--write-cache",
        action="store_true",
        default=False,
        help="Write generated graph sequences to cache",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for cached graph .pt files (default: artifacts/graph_cache)",
    )
    parser.add_argument(
        "--target-name",
        type=str,
        default="partition_any",
        choices=sorted(ALL_TARGETS),
        help="Resilience target column (default: partition_any)",
    )
    parser.add_argument(
        "--experiment-log",
        type=str,
        default="experiments/gnn_log.jsonl",
        help="JSONL experiment log path (default: experiments/gnn_log.jsonl)",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path for metrics JSON (default: next to checkpoint)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use at most this many complete runs after seeded shuffling",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Fast one-command demo mode: one epoch, small hidden dim, smoke paths",
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


def make_run_splits(
    num_samples: int,
    *,
    test_split: float,
    val_split: float,
    seed: int,
    subset: int | None = None,
) -> dict[str, List[int]]:
    """Create seeded train/validation/test splits over complete run indices."""
    if num_samples < 3:
        raise ValueError("Need at least 3 complete runs for train/validation/test splits")
    if not (0.0 < test_split < 1.0):
        raise ValueError("test_split must be in the open interval (0, 1)")
    if not (0.0 <= val_split < 1.0):
        raise ValueError("val_split must be in the interval [0, 1)")
    if test_split + val_split >= 1.0:
        raise ValueError("test_split + val_split must be less than 1")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(num_samples).tolist()
    if subset is not None:
        if subset < 3:
            raise ValueError("subset must be at least 3 when provided")
        shuffled = shuffled[: min(subset, num_samples)]

    n = len(shuffled)
    n_test = max(1, int(round(n * test_split)))
    n_val = max(1, int(round(n * val_split))) if val_split > 0.0 else 0
    if n_test + n_val >= n:
        n_val = max(0, n - n_test - 1)
    if n_test + n_val >= n:
        raise ValueError("Split sizes leave no training samples")

    test_indices = shuffled[:n_test]
    val_indices = shuffled[n_test:n_test + n_val]
    train_indices = shuffled[n_test + n_val:]

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def _safe_rank_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) < 2:
        return float("nan"), float("nan")
    try:
        from scipy.stats import kendalltau, spearmanr
    except ImportError:
        return float("nan"), float("nan")

    spearman_val, _ = spearmanr(y_true, y_pred)
    kendall_val, _ = kendalltau(y_true, y_pred)
    return float(spearman_val), float(kendall_val)


def _regression_metrics(y_true: List[float], y_pred: List[float], loss: float) -> dict[str, Any]:
    n_samples = len(y_true)
    metrics: dict[str, Any] = {
        "loss": float(loss),
        "n_samples": int(n_samples),
        "mae": float("nan"),
        "rmse": float("nan"),
        "r2": float("nan"),
        "spearman": float("nan"),
        "kendall": float("nan"),
    }
    if n_samples == 0:
        return metrics

    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    residual = true_arr - pred_arr

    metrics["mae"] = float(np.mean(np.abs(residual)))
    metrics["rmse"] = float(np.sqrt(np.mean(residual ** 2)))

    if n_samples >= 2:
        ss_res = float(np.sum(residual ** 2))
        mean_true = float(np.mean(true_arr))
        ss_tot = float(np.sum((true_arr - mean_true) ** 2))
        metrics["r2"] = float("nan") if ss_tot == 0.0 else float(1.0 - (ss_res / ss_tot))

    spearman_val, kendall_val = _safe_rank_metrics(true_arr, pred_arr)
    metrics["spearman"] = spearman_val
    metrics["kendall"] = kendall_val
    return metrics


def train_epoch(
    model: SatelliteGNN,
    train_indices: List[int],
    dataset: SatNetTemporalDataset,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy).

    For regression, accuracy is always 0.0 (not meaningful).
    """
    is_regression = model.task_type == "regression"
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for idx in train_indices:
        data_sequence = dataset[idx]
        data_sequence = move_data_to_device(data_sequence, device)
        label = data_sequence[0].y.to(device)  # [1] float

        optimizer.zero_grad()
        out = model(data_sequence)

        if is_regression:
            loss = criterion(out.squeeze(), label)
        else:
            loss = criterion(out, label.long())

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += 1

        if not is_regression:
            pred = out.argmax(dim=1)
            correct += (pred == label.long()).sum().item()

    avg_loss = total_loss / max(len(train_indices), 1)
    accuracy = correct / total if not is_regression else 0.0

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: SatelliteGNN,
    test_indices: List[int],
    dataset: SatNetTemporalDataset,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, dict]:
    """Evaluate the model. Returns (avg_loss, accuracy, metrics_dict)."""
    is_regression = model.task_type == "regression"
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true_reg: List[float] = []
    y_pred_reg: List[float] = []

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for idx in test_indices:
        data_sequence = dataset[idx]
        data_sequence = move_data_to_device(data_sequence, device)
        label = data_sequence[0].y.to(device)

        out = model(data_sequence)

        if is_regression:
            loss = criterion(out.squeeze(), label)
        else:
            loss = criterion(out, label.long())

        total_loss += loss.item()
        total += 1

        if is_regression:
            y_true_reg.append(float(label.item()))
            y_pred_reg.append(float(out.squeeze().item()))
        else:
            pred = out.argmax(dim=1)
            correct += (pred == label.long()).sum().item()

            pred_val = pred.item()
            label_val = label.long().item()
            if pred_val == 1 and label_val == 1:
                true_positives += 1
            elif pred_val == 1 and label_val == 0:
                false_positives += 1
            elif pred_val == 0 and label_val == 0:
                true_negatives += 1
            else:
                false_negatives += 1

    avg_loss = total_loss / max(len(test_indices), 1)
    if is_regression:
        metrics = _regression_metrics(y_true_reg, y_pred_reg, avg_loss)
        accuracy = 0.0
        return avg_loss, accuracy, metrics

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "n_samples": int(total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [
            [true_negatives, false_positives],
            [false_negatives, true_positives],
        ],
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
    }

    return avg_loss, accuracy, metrics


def main():
    """Main training loop."""
    args = parse_args()

    if args.smoke:
        args.epochs = 1
        args.hidden_dim = min(args.hidden_dim, 16)
        args.subset = args.subset or 8
        args.test_split = max(args.test_split, 0.25)
        args.val_split = max(args.val_split, 0.25)
        if args.output_model == DEFAULT_OUTPUT_MODEL:
            args.output_model = "artifacts/smoke/gnn/satellite_gnn_smoke.pt"

    task_type = infer_task_type(args.target_name)
    is_regression = task_type == "regression"

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

    if args.metrics_output is None:
        metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
    else:
        metrics_path = Path(args.metrics_output)
        if not metrics_path.is_absolute():
            metrics_path = PROJECT_ROOT / metrics_path
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.experiment_log)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path

    # Get device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Target: {args.target_name} ({task_type})")

    if SatelliteGNN is None:
        logger.error("Temporal GNN dependency unavailable: %s", GNN_MODEL_IMPORT_ERROR)
        logger.error(
            "Install the optional GNN stack, including torch-geometric-temporal, "
            "before running this training script."
        )
        sys.exit(1)

    with ExperimentLogger(log_path) as exp_log:
        exp_log.set("model_type", "SatelliteGNN")
        exp_log.set("target_name", args.target_name)
        exp_log.set("task_type", task_type)
        exp_log.set("seed", args.seed)
        exp_log.set("data_path", str(data_dir))
        exp_log.set("use_cache", bool(args.use_cache))
        exp_log.set("write_cache", bool(args.write_cache))
        exp_log.set("cache_dir", args.cache_dir)
        exp_log.set("smoke", bool(args.smoke))
        exp_log.set("subset", args.subset)

        # Load dataset
        logger.info(f"Loading dataset from {data_dir}")
        try:
            dataset = SatNetTemporalDataset(
                root=str(data_dir),
                use_cache=args.use_cache,
                write_cache=args.write_cache,
                cache_dir=args.cache_dir,
                target_name=args.target_name,
            )
        except FileNotFoundError as e:
            logger.error(f"Dataset not found: {e}")
            logger.error("Please ensure tier1_design_runs.csv exists in the data directory.")
            sys.exit(1)

        exp_log.set("resolved_cache_dir", dataset._resolved_cache_dir)
        num_samples = len(dataset)
        neg_count, pos_count = dataset.get_label_distribution()
        prediction_join_keys = _collect_prediction_join_keys(dataset)
        logger.info(f"Dataset: {num_samples} samples (neg: {neg_count}, pos: {pos_count})")
        exp_log.set("num_samples", num_samples)

        # Split complete runs; each dataset index is one full graph sequence.
        splits = make_run_splits(
            num_samples,
            test_split=args.test_split,
            val_split=args.val_split,
            seed=args.seed,
            subset=args.subset,
        )
        train_indices = splits["train"]
        val_indices = splits["val"]
        test_indices = splits["test"]
        logger.info(
            "Train/Val/Test split: %d/%d/%d complete runs",
            len(train_indices), len(val_indices), len(test_indices),
        )
        exp_log.set("train_size", len(train_indices))
        exp_log.set("val_size", len(val_indices))
        exp_log.set("test_size", len(test_indices))

        # Initialize model
        out_channels = 1 if is_regression else 2
        model = SatelliteGNN(
            node_features=3,
            hidden_channels=args.hidden_dim,
            out_channels=out_channels,
            task_type=task_type,
        )
        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        exp_log.set("model_parameters", int(num_params))

        # Loss and optimizer
        if is_regression:
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Training loop
        logger.info(f"Starting training for {args.epochs} epochs...")
        logger.info("-" * 60)
        exp_log.start_timer("training")

        best_val_loss = float("inf")
        best_epoch = 0
        best_metrics: dict[str, Any] = {}

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_indices, dataset, optimizer, criterion, device
            )
            val_loss, val_acc, val_metrics = evaluate(
                model, val_indices, dataset, criterion, device
            )

            if is_regression:
                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val MAE: {val_metrics['mae']:.4f} | "
                    f"Val RMSE: {val_metrics['rmse']:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch:3d}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
                    f"F1: {val_metrics['f1']:.4f}"
                )

            # Save best model by validation loss. The test split is held out
            # until after training to avoid using it for checkpoint selection.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "task_type": task_type,
                    "target_name": args.target_name,
                }, output_path)

        exp_log.stop_timer("training")

        checkpoint = torch.load(output_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc, test_metrics = evaluate(
            model, test_indices, dataset, criterion, device
        )

        logger.info("-" * 60)
        logger.info(
            f"Training complete! Best val loss: {best_val_loss:.4f} at epoch {best_epoch}"
        )
        if is_regression:
            logger.info(
                "Held-out test: loss=%.4f MAE=%.4f RMSE=%.4f R2=%.4f",
                test_loss,
                test_metrics["mae"],
                test_metrics["rmse"],
                test_metrics["r2"],
            )
        else:
            logger.info(
                "Held-out test: loss=%.4f acc=%.4f precision=%.4f recall=%.4f f1=%.4f",
                test_loss,
                test_acc,
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1"],
            )
        logger.info(f"Model saved to: {output_path}")

        # Save per-sample predictions
        model.eval()
        preds_rows = []
        with torch.no_grad():
            for split_name, indices in [
                ("train", train_indices),
                ("val", val_indices),
                ("test", test_indices),
            ]:
                for idx in indices:
                    data_seq = dataset[idx]
                    data_seq = move_data_to_device(data_seq, device)
                    out = model(data_seq)
                    if is_regression:
                        pred_val = float(out.squeeze().item())
                    else:
                        pred_val = float(out.argmax(dim=1).item())
                    true_val = float(data_seq[0].y.item())
                    cfg = dataset.get_run_config(idx)

                    row: dict[str, Any] = {
                        "config_hash": prediction_join_keys[int(idx)],
                        "target_name": args.target_name,
                        "task_type": task_type,
                        "seed": args.seed,
                        "split": split_name,
                        "sample_idx": int(idx),
                        "y_true": true_val,
                        "y_pred": pred_val,
                        "model_type": "SatelliteGNN",
                        "data_path": str(data_dir),
                    }
                    if "run_id" in cfg and not pd.isna(cfg.get("run_id")):
                        row["run_id"] = int(cfg["run_id"])
                    preds_rows.append(row)

        preds_path = output_path.parent / f"{output_path.stem}_predictions.csv"
        pd.DataFrame(preds_rows).to_csv(preds_path, index=False)
        logger.info(f"Predictions saved to: {preds_path}")

        metrics_payload = {
            "model_type": "SatelliteGNN",
            "target_name": args.target_name,
            "task_type": task_type,
            "seed": args.seed,
            "data_dir": str(data_dir),
            "num_samples": int(num_samples),
            "subset": args.subset,
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "test_size": len(test_indices),
            "split_strategy": "seeded_complete_run_indices",
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "best_val_metrics": best_metrics,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_metrics": test_metrics,
            "model_path": str(output_path),
            "prediction_path": str(preds_path),
        }
        with metrics_path.open("w") as f:
            import json
            json.dump(metrics_payload, f, indent=2, default=str)
        logger.info(f"Metrics saved to: {metrics_path}")

        exp_log.set("model_path", str(output_path))
        exp_log.set("prediction_path", str(preds_path))
        exp_log.set("metrics_path", str(metrics_path))
        exp_log.set("best_epoch", best_epoch)
        exp_log.set("best_val_loss", float(best_val_loss))
        exp_log.set("test_loss", float(test_loss))
        exp_log.set_metrics({f"best_val_{k}": v for k, v in best_metrics.items()})
        exp_log.set_metrics({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == "__main__":
    main()
