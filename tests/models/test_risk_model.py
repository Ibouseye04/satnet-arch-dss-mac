"""Tests for satnet.models.risk_model module.

These tests require sklearn to be installed. Use pytest.importorskip
to skip tests when sklearn is unavailable.
"""

from __future__ import annotations

import pytest

sklearn = pytest.importorskip("sklearn")
pd = pytest.importorskip("pandas")


class TestRiskModelConfig:
    """Tests for RiskModelConfig dataclass."""

    def test_default_values(self) -> None:
        """RiskModelConfig has expected default values."""
        from satnet.models.risk_model import RiskModelConfig

        cfg = RiskModelConfig()
        assert cfg.test_size == 0.2
        assert cfg.random_state == 42
        assert cfg.n_estimators == 200
        assert cfg.max_depth is None

    def test_custom_values(self) -> None:
        """RiskModelConfig accepts custom values."""
        from satnet.models.risk_model import RiskModelConfig

        cfg = RiskModelConfig(test_size=0.3, n_estimators=100, max_depth=10)
        assert cfg.test_size == 0.3
        assert cfg.n_estimators == 100
        assert cfg.max_depth == 10


class TestFeatureColumns:
    """Tests for feature column constants."""

    def test_tier1_v1_feature_columns(self) -> None:
        """TIER1_V1_FEATURE_COLUMNS contains expected columns."""
        from satnet.models.risk_model import TIER1_V1_FEATURE_COLUMNS

        assert "num_planes" in TIER1_V1_FEATURE_COLUMNS
        assert "sats_per_plane" in TIER1_V1_FEATURE_COLUMNS
        assert "altitude_km" in TIER1_V1_FEATURE_COLUMNS
        assert "inclination_deg" in TIER1_V1_FEATURE_COLUMNS

    def test_tier1_v1_design_feature_columns(self) -> None:
        """TIER1_V1_DESIGN_FEATURE_COLUMNS excludes failure params."""
        from satnet.models.risk_model import (
            TIER1_V1_DESIGN_FEATURE_COLUMNS,
            TIER1_V1_FEATURE_COLUMNS,
        )

        # Design features should not include failure probabilities
        assert "node_failure_prob" not in TIER1_V1_DESIGN_FEATURE_COLUMNS
        assert "edge_failure_prob" not in TIER1_V1_DESIGN_FEATURE_COLUMNS

        # Should be a subset of full feature columns
        assert len(TIER1_V1_DESIGN_FEATURE_COLUMNS) < len(TIER1_V1_FEATURE_COLUMNS)


class TestModelSerialization:
    """Tests for save_model and load_model functions."""

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        """Model can be saved and loaded back."""
        from sklearn.ensemble import RandomForestClassifier
        from satnet.models.risk_model import save_model, load_model

        # Create a simple trained model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 0])

        # Save and load
        model_path = tmp_path / "test_model.joblib"
        save_model(model, model_path)
        loaded = load_model(model_path)

        # Verify predictions match
        X_test = [[1, 2], [3, 4]]
        assert list(loaded.predict(X_test)) == list(model.predict(X_test))

    def test_save_creates_parent_dirs(self, tmp_path) -> None:
        """save_model creates parent directories if needed."""
        from sklearn.ensemble import RandomForestClassifier
        from satnet.models.risk_model import save_model

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit([[1], [2]], [0, 1])

        # Save to nested path that doesn't exist
        model_path = tmp_path / "nested" / "dirs" / "model.joblib"
        save_model(model, model_path)

        assert model_path.exists()


class TestRfPredictionSchema:
    """Tests for stable RF prediction export fields."""

    def test_train_rf_model_predictions_include_stable_columns(self, tmp_path) -> None:
        from satnet.models.risk_model import RiskModelConfig, train_rf_model

        rows = []
        for i in range(20):
            rows.append(
                {
                    "run_id": i,
                    "config_hash": f"cfg-{i:04d}",
                    "num_planes": 4 + (i % 3),
                    "sats_per_plane": 6 + (i % 2),
                    "total_satellites": 24 + i,
                    "inclination_deg": 53.0 + (i % 5),
                    "altitude_km": 550.0 + i,
                    "node_failure_prob": 0.01 * (i % 4),
                    "edge_failure_prob": 0.02 * (i % 3),
                    "duration_minutes": 10,
                    "step_seconds": 60,
                    "partition_any": i % 2,
                }
            )
        csv_path = tmp_path / "tier1_design_runs.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        cfg = RiskModelConfig(test_size=0.25, random_state=42, n_estimators=10)
        _, _, predictions = train_rf_model(
            csv_path=csv_path,
            target_name="partition_any",
            cfg=cfg,
        )

        required_columns = {
            "config_hash",
            "target_name",
            "task_type",
            "seed",
            "split",
            "sample_idx",
            "y_true",
            "y_pred",
        }
        assert required_columns.issubset(predictions.columns)
        assert set(predictions["target_name"]) == {"partition_any"}
        assert set(predictions["task_type"]) == {"classification"}

    @pytest.mark.parametrize("bad_config_hash", [None, "", "   "])
    def test_train_rf_model_rejects_invalid_config_hash_for_prediction_export(
        self,
        tmp_path,
        bad_config_hash,
    ) -> None:
        from satnet.models.risk_model import RiskModelConfig, train_rf_model

        rows = []
        for i in range(20):
            rows.append(
                {
                    "run_id": i,
                    "config_hash": f"cfg-{i:04d}",
                    "num_planes": 4 + (i % 3),
                    "sats_per_plane": 6 + (i % 2),
                    "total_satellites": 24 + i,
                    "inclination_deg": 53.0 + (i % 5),
                    "altitude_km": 550.0 + i,
                    "node_failure_prob": 0.01 * (i % 4),
                    "edge_failure_prob": 0.02 * (i % 3),
                    "duration_minutes": 10,
                    "step_seconds": 60,
                    "partition_any": i % 2,
                }
            )
        rows[3]["config_hash"] = bad_config_hash
        csv_path = tmp_path / "tier1_design_runs.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        cfg = RiskModelConfig(test_size=0.25, random_state=42, n_estimators=10)
        with pytest.raises(ValueError, match="stable ranking joins"):
            train_rf_model(
                csv_path=csv_path,
                target_name="partition_any",
                cfg=cfg,
            )
