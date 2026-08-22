"""Execute the explicit adaptive-v2 ML export profile.

This entrypoint reads the accepted persisted adaptive production artifacts and
writes only ML representations. It never runs simulation or model training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from export_final_ml_datasets import ADAPTIVE_V2_EXPORT_PROFILE, Exporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, default=ADAPTIVE_V2_EXPORT_PROFILE.source_root)
    parser.add_argument("--replay-root", type=Path, default=ADAPTIVE_V2_EXPORT_PROFILE.replay_root)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, default=ADAPTIVE_V2_EXPORT_PROFILE.contract_root)
    parser.add_argument("--schema-root", type=Path, default=ADAPTIVE_V2_EXPORT_PROFILE.schema_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Exporter(
        production_root=args.production_root,
        replay_root=args.replay_root,
        acceptance_root=Path("."),
        audit_root=Path("."),
        ml_contract_root=args.schema_root,
        contract_root=args.contract_root,
        output_root=args.output_root,
        profile=ADAPTIVE_V2_EXPORT_PROFILE,
    ).run()


if __name__ == "__main__":
    main()
