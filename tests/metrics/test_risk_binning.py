"""Tests for satnet.metrics.risk_binning module."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")


class TestBinSatelliteRisk:
    """Tests for bin_satellite_risk function."""

    def test_bins_scores_correctly(self) -> None:
        """Scores are binned into correct tiers based on thresholds."""
        from satnet.metrics.risk_binning import (
            bin_satellite_risk,
            TIER_HEALTHY,
            TIER_WATCHLIST,
            TIER_CRITICAL,
        )

        df = pd.DataFrame({
            "reliability_score": [0.95, 0.7, 0.3, 0.81, 0.5, 0.49]
        })
        result = bin_satellite_risk(df)

        # Default thresholds: healthy > 0.8, critical < 0.5
        expected_tiers = [
            TIER_HEALTHY,    # 0.95 > 0.8
            TIER_WATCHLIST,  # 0.5 <= 0.7 <= 0.8
            TIER_CRITICAL,   # 0.3 < 0.5
            TIER_HEALTHY,    # 0.81 > 0.8
            TIER_WATCHLIST,  # 0.5 <= 0.5 <= 0.8
            TIER_CRITICAL,   # 0.49 < 0.5
        ]
        assert result["risk_tier"].tolist() == expected_tiers

    def test_adds_label_and_action_columns(self) -> None:
        """Result includes risk_tier, risk_label, and recommended_action columns."""
        from satnet.metrics.risk_binning import bin_satellite_risk

        df = pd.DataFrame({"reliability_score": [0.9]})
        result = bin_satellite_risk(df)

        assert "risk_tier" in result.columns
        assert "risk_label" in result.columns
        assert "recommended_action" in result.columns
        assert result["risk_label"].iloc[0] == "Healthy"
        assert result["recommended_action"].iloc[0] == "No Action"

    def test_custom_thresholds(self) -> None:
        """Custom thresholds are applied correctly."""
        from satnet.metrics.risk_binning import (
            bin_satellite_risk,
            TIER_HEALTHY,
            TIER_CRITICAL,
        )

        df = pd.DataFrame({"reliability_score": [0.95, 0.85, 0.75]})
        result = bin_satellite_risk(df, healthy_threshold=0.9, critical_threshold=0.8)

        # With new thresholds: healthy > 0.9, critical < 0.8
        assert result["risk_tier"].iloc[0] == TIER_HEALTHY    # 0.95 > 0.9
        assert result["risk_tier"].iloc[2] == TIER_CRITICAL   # 0.75 < 0.8

    def test_raises_for_missing_column(self) -> None:
        """Raises ValueError when score column is missing."""
        from satnet.metrics.risk_binning import bin_satellite_risk

        df = pd.DataFrame({"other_column": [0.5]})
        with pytest.raises(ValueError, match="not found"):
            bin_satellite_risk(df)

    def test_raises_for_invalid_thresholds(self) -> None:
        """Raises ValueError when thresholds are inverted."""
        from satnet.metrics.risk_binning import bin_satellite_risk

        df = pd.DataFrame({"reliability_score": [0.5]})
        with pytest.raises(ValueError, match="Invalid thresholds"):
            bin_satellite_risk(df, healthy_threshold=0.3, critical_threshold=0.7)

    def test_raises_for_out_of_range_scores(self) -> None:
        """Raises ValueError when scores are outside [0, 1]."""
        from satnet.metrics.risk_binning import bin_satellite_risk

        df = pd.DataFrame({"reliability_score": [1.5]})
        with pytest.raises(ValueError, match="outside"):
            bin_satellite_risk(df)

    def test_raises_for_nan_scores(self) -> None:
        """Raises ValueError when scores contain NaN."""
        from satnet.metrics.risk_binning import bin_satellite_risk
        import math

        df = pd.DataFrame({"reliability_score": [0.5, float("nan")]})
        with pytest.raises(ValueError, match="NaN"):
            bin_satellite_risk(df)


class TestTierConstants:
    """Tests for tier constant values."""

    def test_tier_values(self) -> None:
        """Tier constants have expected values."""
        from satnet.metrics.risk_binning import (
            TIER_HEALTHY,
            TIER_WATCHLIST,
            TIER_CRITICAL,
        )

        assert TIER_HEALTHY == 1
        assert TIER_WATCHLIST == 2
        assert TIER_CRITICAL == 3

    def test_tier_labels(self) -> None:
        """TIER_LABELS maps tiers to human-readable names."""
        from satnet.metrics.risk_binning import (
            TIER_LABELS,
            TIER_HEALTHY,
            TIER_WATCHLIST,
            TIER_CRITICAL,
        )

        assert TIER_LABELS[TIER_HEALTHY] == "Healthy"
        assert TIER_LABELS[TIER_WATCHLIST] == "Watchlist"
        assert TIER_LABELS[TIER_CRITICAL] == "Critical"
