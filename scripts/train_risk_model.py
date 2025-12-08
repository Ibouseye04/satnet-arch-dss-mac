from pathlib import Path
import sys
import json

# --- Make sure Python can see the `src` folder ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satnet.models.risk_model import (  # noqa: E402
    RiskModelConfig,
    train_risk_model,
    save_model,
)


def main() -> None:
    data_path = PROJECT_ROOT / "Data" / "failure_dataset.csv"
    model_path = PROJECT_ROOT / "models" / "risk_model.joblib"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Run scripts/export_failure_dataset.py first."
        )

    cfg = RiskModelConfig(
        test_size=0.2,
        random_state=42,
        n_estimators=300,
        max_depth=None,
    )

    print(f"Loading dataset from {data_path} ...")
    model, metrics = train_risk_model(data_path, cfg)

    save_model(model, model_path)
    print(f"Saved model to {model_path}")

    # pretty-print some key metrics
    print("=== Evaluation Metrics (test split) ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")

    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(metrics["confusion_matrix"])

    print("\nTop feature importances:")
    fi = metrics["feature_importances"]
    for k, v in sorted(fi.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v:.3f}")

    # optional: dump full metrics JSON for later analysis
    metrics_path = PROJECT_ROOT / "models" / "risk_model_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFull metrics written to {metrics_path}")


if __name__ == "__main__":
    main()