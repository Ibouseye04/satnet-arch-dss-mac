from __future__ import annotations

import math
from pathlib import Path

import pytest

import satnet.network.hypatia_adapter as hypatia_adapter


class _FailingSatellite:
    def sgp4(self, jd: float, fr: float):
        return 6, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0)


class _FailingSatrecBoundary:
    @staticmethod
    def twoline2rv(line1: str, line2: str, gravity_model):
        return _FailingSatellite()


def test_v02_nonzero_sgp4_error_fails_closed_without_partial_isl_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(hypatia_adapter, "Satrec", _FailingSatrecBoundary)

    with hypatia_adapter.HypatiaAdapter(
        num_planes=2,
        sats_per_plane=3,
        output_dir=tmp_path,
    ) as adapter:
        adapter.generate_tles()
        with pytest.raises(hypatia_adapter.SGP4PropagationError) as exc_info:
            adapter.calculate_isls(duration_minutes=0, step_seconds=60)

        error = exc_info.value
        assert error.sat_id == 0
        assert error.target_time == adapter.config.epoch
        assert error.error_code == 6
        assert "satellite 0" in str(error)
        assert "error code 6" in str(error)
        assert adapter._isl_data == {}

    assert not (tmp_path / "isls.txt").exists()


def test_v02_canonical_adapter_requires_available_sgp4(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(hypatia_adapter, "SGP4_AVAILABLE", False)

    with pytest.raises(hypatia_adapter.OrbitalEngineUnavailableError) as exc_info:
        hypatia_adapter.HypatiaAdapter(output_dir=tmp_path)

    assert exc_info.value.orbital_engine == hypatia_adapter.ORBITAL_ENGINE_SGP4
    assert "canonical execution requires SGP4" in str(exc_info.value)


def test_v02_keplerian_engine_requires_explicit_noncanonical_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(hypatia_adapter, "SGP4_AVAILABLE", False)

    with hypatia_adapter.HypatiaAdapter(
        num_planes=2,
        sats_per_plane=3,
        output_dir=tmp_path,
        orbital_engine=hypatia_adapter.ORBITAL_ENGINE_KEPLERIAN,
    ) as adapter:
        output_path, _ = adapter.calculate_isls(duration_minutes=0, step_seconds=60)
        positions = adapter.get_positions_at_step(0)

    assert output_path.exists()
    assert len(positions) == 6
    assert all(
        math.isfinite(coordinate)
        for position in positions
        for coordinate in (position.x_km, position.y_km, position.z_km, position.alt_km)
    )
    assert all(
        (position.x_km, position.y_km, position.z_km) != (0.0, 0.0, 0.0)
        for position in positions
    )
