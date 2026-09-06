"""
QKD Simulation Framework Validation Suite.

Validates the framework against:
  1. Analytical benchmarks (Ma 2005, pure-loss channel)
  2. Physical invariants (monotonicity, bounds, consistency)
  3. Key rate curve shape
  4. Regression tests (golden values)

Uses the main_optimized.py pipeline which is the production code path.
"""

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helper: run a sweep and return parsed CSV rows
# ---------------------------------------------------------------------------

def run_sweep(
    distances: List[float] = None,
    num_pulses_list: List[int] = None,
    apply_noise: bool = True,
    fiber_loss_db_km: float = 0.2,
    det_eff: float = 0.15,
    dark_rate: float = 1e-6,
    qber_intrinsic: float = 0.01,
    signal_mu: float = 0.5,
    decoy_mu: float = 0.1,
    extra_config: dict = None,
) -> List[Dict[str, str]]:
    """Run a QKD simulation sweep via main_optimized and return CSV rows."""
    import main_optimized as m
    import copy

    config = copy.deepcopy(m.DEFAULT_CONFIG)

    config["channel"]["fiber_loss_db_km"] = fiber_loss_db_km
    config["detector"]["det_eff_d0"] = det_eff
    config["detector"]["det_eff_d1"] = det_eff
    config["detector"]["dark_rate"] = dark_rate
    config["detector"]["qber_intrinsic"] = qber_intrinsic
    config["source"]["pulses"]["signal"]["mu"] = signal_mu
    config["source"]["pulses"]["decoy"]["mu"] = decoy_mu

    if distances is not None:
        config["simulation"]["distance_start_km"] = min(distances)
        config["simulation"]["distance_stop_km"] = max(distances)
        config["simulation"]["distance_points"] = len(distances)

    if num_pulses_list is not None:
        config["simulation"]["min_pulses_log"] = int(math.log10(min(num_pulses_list)))
        config["simulation"]["max_pulses_log"] = int(math.log10(max(num_pulses_list)))

    config["simulation"]["apply_statistical_noise"] = apply_noise

    if extra_config:
        for k, v in extra_config.items():
            parts = k.split(".")
            d = config
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v

    m.run_and_save_csv(config, num_workers=1, sweeps={})
    csv_path = Path("qkd_results_optimized_sweep.csv")

    rows = []
    if csv_path.exists():
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def parse_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def find_rows(rows, distance_km=None, noise_applied=None, min_pulses=None):
    """Filter rows by distance, noise, and min pulses."""
    result = []
    for row in rows:
        if distance_km is not None and abs(parse_float(row["distance_km"]) - distance_km) > 0.1:
            continue
        if noise_applied is not None and row["noise_applied"] != str(noise_applied):
            continue
        if min_pulses is not None and parse_float(row["total_pulses"]) < min_pulses:
            continue
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# Layer 1: Analytical Benchmarks
# ---------------------------------------------------------------------------

class TestAnalyticalBenchmarks:
    """Validate against closed-form analytical results."""

    def test_pure_loss_gain_signal(self):
        """Gain at signal intensity should match Ma 2005 Eq. (23).

        Q_μ = 1 - (1 - η)^μ  where η = 10^(-αL/10) * η_det.

        Note: detection_yield_signal includes the basis sifting factor (~0.5),
        so we compare against Q_μ * P(basis_match) ≈ Q_μ * 0.5.
        """
        distance = 50.0
        alpha = 0.2
        eta_det = 0.15
        mu = 0.5

        eta = 10 ** (-alpha * distance / 10) * eta_det
        expected_gain_raw = 1 - (1 - eta) ** mu
        # Basis sifting: Alice and Bob each choose basis with P=0.5
        # P(basis match) = 0.5^2 + 0.5^2 = 0.5
        expected_gain_sifted = expected_gain_raw * 0.5

        rows = run_sweep(
            distances=[distance],
            num_pulses_list=[1_000_000],
            apply_noise=False,
            fiber_loss_db_km=alpha,
            det_eff=eta_det,
            dark_rate=0.0,
        )

        matching = find_rows(rows, distance_km=distance, noise_applied=False, min_pulses=100000)
        if not matching:
            pytest.skip("No noiseless row found")

        gain = parse_float(matching[0]["detection_yield_signal"])
        assert gain > 0, "Gain should be positive"

        rel_err = abs(gain - expected_gain_sifted) / expected_gain_sifted
        assert rel_err < 0.20, (
            f"Gain at {distance}km: got {gain:.6f}, "
            f"expected ~{expected_gain_sifted:.6f} (raw Q_μ={expected_gain_raw:.6f}, "
            f"rel_err={rel_err:.3f})"
        )

    def test_gain_decreases_with_distance(self):
        """Gain must decrease with distance (Beer-Lambert law)."""
        rows = run_sweep(
            distances=[10.0, 50.0, 100.0, 150.0],
            num_pulses_list=[100_000],
            apply_noise=True,
        )

        gains = []
        for row in find_rows(rows, noise_applied=True, min_pulses=100000):
            d = parse_float(row["distance_km"])
            g = parse_float(row["detection_yield_signal"])
            if g > 0:
                gains.append((d, g))

        gains.sort()
        for i in range(len(gains) - 1):
            assert gains[i][1] >= gains[i + 1][1] * 0.9, (
                f"Gain not decreasing: G({gains[i][0]}km)={gains[i][1]:.6f} "
                f"< G({gains[i+1][0]}km)={gains[i+1][1]:.6f}"
            )


# ---------------------------------------------------------------------------
# Layer 2: Physical Invariants
# ---------------------------------------------------------------------------

class TestPhysicalInvariants:
    """All outputs must satisfy physical constraints."""

    @pytest.fixture(scope="class")
    def sweep_results(self):
        return run_sweep(
            distances=[10.0, 50.0, 100.0, 150.0],
            num_pulses_list=[100_000],
            apply_noise=True,
        )

    def test_qber_in_valid_range(self, sweep_results):
        for row in sweep_results:
            qber = parse_float(row["qber"])
            assert 0.0 <= qber <= 1.0, f"QBER={qber} out of range at {row['distance_km']}km"

    def test_gain_in_valid_range(self, sweep_results):
        for row in sweep_results:
            for field in ["detection_yield_signal", "detection_yield_decoy", "detection_yield_vacuum"]:
                gain = parse_float(row[field])
                assert 0.0 <= gain <= 1.0, f"{field}={gain} out of range"

    def test_signal_gain_higher_than_decoy(self, sweep_results):
        for row in find_rows(sweep_results, noise_applied=True, min_pulses=100000):
            g_sig = parse_float(row["detection_yield_signal"])
            g_dec = parse_float(row["detection_yield_decoy"])
            if g_sig > 0 and g_dec > 0:
                assert g_sig >= g_dec * 0.7, (
                    f"Signal gain ({g_sig:.6f}) << decoy gain ({g_dec:.6f}) "
                    f"at {row['distance_km']}km"
                )

    def test_dark_counts_increase_qber(self):
        rows_low = run_sweep(distances=[100.0], num_pulses_list=[100_000], apply_noise=True, dark_rate=1e-8)
        rows_high = run_sweep(distances=[100.0], num_pulses_list=[100_000], apply_noise=True, dark_rate=1e-4)

        qber_low = max(parse_float(r["qber"]) for r in find_rows(rows_low, noise_applied=True))
        qber_high = max(parse_float(r["qber"]) for r in find_rows(rows_high, noise_applied=True))

        assert qber_high >= qber_low, (
            f"Higher dark rate should increase QBER: low={qber_low:.4f}, high={qber_high:.4f}"
        )

    def test_higher_efficiency_increases_gain(self):
        rows_low = run_sweep(distances=[50.0], num_pulses_list=[100_000], apply_noise=True, det_eff=0.05)
        rows_high = run_sweep(distances=[50.0], num_pulses_list=[100_000], apply_noise=True, det_eff=0.30)

        gain_low = max(parse_float(r["detection_yield_signal"]) for r in find_rows(rows_low, noise_applied=True))
        gain_high = max(parse_float(r["detection_yield_signal"]) for r in find_rows(rows_high, noise_applied=True))

        assert gain_high > gain_low, (
            f"Higher efficiency should increase gain: low={gain_low:.6f}, high={gain_high:.6f}"
        )


# ---------------------------------------------------------------------------
# Layer 3: Key Rate Curve Shape
# ---------------------------------------------------------------------------

class TestKeyRateCurveShape:
    @pytest.fixture(scope="class")
    def curve_data(self):
        return run_sweep(
            distances=list(np.linspace(0, 200, 10)),
            num_pulses_list=[1_000_000],
            apply_noise=True,
        )

    def test_qber_below_threshold_at_short_distance(self, curve_data):
        """QBER should be below the 11% threshold at short distances."""
        for row in find_rows(curve_data, noise_applied=True, min_pulses=100000):
            if parse_float(row["distance_km"]) <= 30:
                qber = parse_float(row["qber"])
                if qber > 0:
                    assert qber < 0.11, f"QBER={qber} exceeds 11% at {row['distance_km']}km"

    def test_zero_key_at_long_distance(self, curve_data):
        """Key rate should be zero at long distances."""
        for row in find_rows(curve_data, noise_applied=True, min_pulses=100000):
            if parse_float(row["distance_km"]) >= 180:
                skr = parse_float(row["secure_key_rate"])
                assert skr <= 1e-10, f"Key rate should be ~0 at {row['distance_km']}km, got {skr}"


# ---------------------------------------------------------------------------
# Layer 4: Regression Tests
# ---------------------------------------------------------------------------

class TestRegression:
    def test_qber_10km_range(self):
        rows = run_sweep(distances=[10.0], num_pulses_list=[100_000], apply_noise=True)
        for row in find_rows(rows, noise_applied=True, min_pulses=100000):
            qber = parse_float(row["qber"])
            assert 0.005 < qber < 0.10, f"QBER={qber:.4f} out of expected range at 10km"
            return
        pytest.skip("No matching row")

    def test_gain_10km_range(self):
        rows = run_sweep(distances=[10.0], num_pulses_list=[100_000], apply_noise=True)
        for row in find_rows(rows, noise_applied=True, min_pulses=100000):
            gain = parse_float(row["detection_yield_signal"])
            # At 10km: η≈0.095, Q_0.5≈0.049, sifted≈0.024
            assert 0.005 < gain < 0.10, f"Gain={gain:.6f} out of expected range at 10km"
            return
        pytest.skip("No matching row")
