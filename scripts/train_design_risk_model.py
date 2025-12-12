from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from satnet.models.risk_model import (  # noqa: E402
    RiskModelConfig,
    train_tier1_risk_model,
    save_model,
)


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "design_dataset_tier1.csv"
    model_path = PROJECT_ROOT / "models" / "design_risk_model_tier1.joblib"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            "Ensure the tier1 dataset has been generated."
        )

    cfg = RiskModelConfig(
        test_size=0.2,
        random_state=42,
        n_estimators=300,
        max_depth=None,
    )

    print(f"Loading tier1 design dataset from {data_path} ...")
    model, metrics = train_tier1_risk_model(data_path, cfg)

    save_model(model, model_path)
    print(f"Saved tier1 design-time model to {model_path}")

    print("\n=== Classification Report ===")
    cr = metrics["classification_report"]
    print(f"              precision    recall  f1-score   support")
    for label in ["0", "1"]:
        if label in cr:
            r = cr[label]
            print(f"    {label:>8}      {r['precision']:.2f}      {r['recall']:.2f}      {r['f1-score']:.2f}      {int(r['support'])}")
    print(f"\n    accuracy                          {cr['accuracy']:.2f}      {int(cr['macro avg']['support'])}")
    print(f"   macro avg      {cr['macro avg']['precision']:.2f}      {cr['macro avg']['recall']:.2f}      {cr['macro avg']['f1-score']:.2f}      {int(cr['macro avg']['support'])}")
    print(f"weighted avg      {cr['weighted avg']['precision']:.2f}      {cr['weighted avg']['recall']:.2f}      {cr['weighted avg']['f1-score']:.2f}      {int(cr['weighted avg']['support'])}")

    print("\n=== Tier1 Design-Time Model Metrics (test split) ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(metrics["confusion_matrix"])

    print("\nTop design feature importances:")
    fi = metrics["feature_importances"]
    for k, v in sorted(fi.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v:.3f}")

    metrics_path = PROJECT_ROOT / "models" / "design_risk_model_tier1_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nFull metrics written to {metrics_path}")


if __name__ == "__main__":
    main()