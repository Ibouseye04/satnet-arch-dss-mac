from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pandas as pd  # noqa: E402
from satnet.models.risk_model import (  # noqa: E402
    FEATURE_COLUMNS,
    load_model,
    predict_partition_probabilities,
)


def main() -> None:
    model_path = PROJECT_ROOT / "models" / "risk_model.joblib"
    data_path = PROJECT_ROOT / "Data" / "failure_dataset.csv"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run scripts/train_risk_model.py first."
        )

    df = pd.read_csv(data_path)

    # take first 5 scenarios as examples
    X = df[FEATURE_COLUMNS].iloc[:5].copy()

    model = load_model(model_path)
    probs = predict_partition_probabilities(model, X)

    print("=== Example risk scores for first 5 scenarios ===")
    for i, p in enumerate(probs):
        row = df.iloc[i]
        print(
            f"run_id={int(row['run_id'])}, "
            f"failed_nodes={row['failed_nodes']}, "
            f"failed_edges={row['failed_edges']}, "
            f"largest_component_ratio={row['largest_component_ratio']:.3f}, "
            f"true_label={int(row['partitioned'])}, "
            f"predicted_prob={p:.3f}"
        )


if __name__ == "__main__":
    main()