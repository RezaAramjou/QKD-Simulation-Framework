"""Physics-validation tests for the SPAD detector module.

These tests pin down the two detector-modeling fixes identified by the
analytical validation of the decoy-state BB84 simulator:

1. **Geiger-mode guard** -- an SPAD only avalanches when ``bias_voltage``
   strictly exceeds ``breakdown_voltage``.  An invalid (non-Geiger) operating
   point must raise loudly instead of being silently clamped (which previously
   masked the misconfiguration behind a zero dark rate).

2. **Standard SPAD click physics** -- in the event-driven path a threshold
   SPAD triggers deterministically whenever at least one photon reaches it
   (after channel thinning), with detector quantum efficiency applied exactly
   once downstream.  This removes the former redundant ``1 - exp(-mu)``
   Bernoulli that double-thinned already-sampled integer arrivals and capped
   single-photon clicks at ``1 - exp(-1) ~= 0.6321`` even at unity
   transmission.

Run from the ``qkd/`` directory so the real ``qkd`` package (and
``main_optimized`` at the framework root) are importable::

    cd qkd && python -m pytest tests/test_physics_validation.py -v
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

# Make the framework root (parent of ``qkd/``) importable so that
# ``main_optimized`` can be imported for the build_detector guard test.
_FRAMEWORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_ROOT)

from qkd.datatypes import DetectorType, DoubleClickPolicy
from qkd.detectors import SinglePhotonDetector


# ---------------------------------------------------------------------------
# Test 1: Geiger-mode guard
# ---------------------------------------------------------------------------

def _geiger_detector(*, bias_voltage: float, breakdown_voltage: float) -> SinglePhotonDetector:
    """Construct a minimal SPAD with the given bias / breakdown voltages."""
    return SinglePhotonDetector(
        det_eff_d0=1.0,
        det_eff_d1=1.0,
        dark_rate=1e-6,
        qber_intrinsic=0.0,
        misalignment=0.0,
        double_click_policy=DoubleClickPolicy.RANDOM,
        detector_type=DetectorType.SPD,
        dead_time_ns=0.0,
        jitter_fwhm_ns=0.0,
        afterpulse_prob=0.0,
        bias_voltage=bias_voltage,
        breakdown_voltage=breakdown_voltage,
        ref_bias_voltage=50.0,
    )


def test_geiger_mode_guard_rejects_bias_below_breakdown():
    """bias_voltage (44 V) below breakdown_voltage (45 V) must raise ValueError.

    Regression test for the former silent-clamp behaviour: the detector used
    to clamp to ``breakdown_voltage - 1`` and return a zero dark rate,
    masquerading as a noiseless detector rather than a non-operational one.
    """
    with pytest.raises(ValueError, match="Geiger mode"):
        _geiger_detector(bias_voltage=44.0, breakdown_voltage=45.0)


def test_geiger_mode_guard_rejects_bias_equal_to_breakdown():
    """The boundary case (bias == breakdown) is also non-Geiger and must raise."""
    with pytest.raises(ValueError):
        _geiger_detector(bias_voltage=45.0, breakdown_voltage=45.0)


def test_geiger_mode_guard_accepts_valid_overbias():
    """A valid overbias configuration must construct without error."""
    det = _geiger_detector(bias_voltage=50.0, breakdown_voltage=45.0)
    assert det.bias_voltage > det.breakdown_voltage


def test_build_detector_geiger_mode_guard():
    """The main_optimized.build_detector() entry point must enforce the same
    guard, since it sets the overbias attributes via setattr *after*
    construction (bypassing ``_validate_params``)."""
    import main_optimized  # noqa: WPS433 -- deferred import keeps qkd-only runs working

    config = {
        "detector": {
            "det_eff_d0": 1.0,
            "det_eff_d1": 1.0,
            "dark_rate": 1e-6,
            "bias_voltage": 44.0,
            "breakdown_voltage": 45.0,
            "ref_bias_voltage": 50.0,
        }
    }
    with pytest.raises(ValueError, match="Geiger mode"):
        main_optimized.build_detector(config)


# ---------------------------------------------------------------------------
# Test 2: Standard SPAD yield (no artificial 0.6321 attenuation)
# ---------------------------------------------------------------------------

def _noiseless_detector() -> SinglePhotonDetector:
    """A clean SPAD: unit efficiency, no dark counts, no jitter/dead-time/afterpulse.

    With ``noise_applied=False`` semantics, the only contribution to a click is
    the signal path, so the per-pulse click probability reduces to the standard
    Poisson prediction ``P_click = 1 - exp(-eta_det * eff_trans * mu)``.
    """
    return SinglePhotonDetector(
        det_eff_d0=1.0,
        det_eff_d1=1.0,
        dark_rate=0.0,           # no dark counts
        qber_intrinsic=0.0,      # no intrinsic bit errors
        misalignment=0.0,        # no basis misalignment flips
        double_click_policy=DoubleClickPolicy.RANDOM,
        detector_type=DetectorType.SPD,
        dead_time_ns=0.0,        # no dead-time suppression
        jitter_fwhm_ns=0.0,      # deterministic timing -> clean pulse binning
        afterpulse_prob=0.0,     # no afterpulses
        bias_voltage=50.0,
        breakdown_voltage=45.0,
        ref_bias_voltage=50.0,
    )


def _signal_yield(detector: SinglePhotonDetector, photon_numbers: np.ndarray, rng: np.random.Generator) -> float:
    """Run one clean detection pass and return the observed click yield."""
    num_pulses = len(photon_numbers)
    ideal_outcomes_d0 = np.ones(num_pulses, dtype=bool)
    result = detector.simulate_detection(
        channel_transmittance=1.0,   # eta_ch = 1.0 -> no channel loss
        photon_numbers=photon_numbers,
        rng=rng,
        ideal_outcomes_d0=ideal_outcomes_d0,
        pulse_period_ns=100.0,
        return_diagnostics=False,
        # Disable intermodulation (IMD) noise so only the signal path contributes.
        N_channels=1,
        modulation_index=0.0,
        mu_signal=0.0,
    )
    clicks = np.asarray(result.click0) | np.asarray(result.click1)
    return float(np.count_nonzero(clicks)) / float(num_pulses)


def test_standard_spad_yield_matches_poisson_prediction():
    """Signal yield must converge to ``1 - exp(-mu)`` for a noiseless SPAD at
    unity channel/detector efficiency.

    The former ``sig_mask = rng.random() < (1 - exp(-mu_sig))`` Bernoulli
    double-thinned the already-sampled integer photon numbers, yielding
    ``1 - exp(-mu * (1 - 1/e))`` instead -- an artificial ``1 - 1/e ~= 0.6321``
    attenuation of the effective mean photon number.  This test asserts the
    yield matches the standard Poisson prediction and is *not* consistent with
    the attenuated (buggy) prediction.
    """
    rng = np.random.default_rng(20260725)
    mu = 0.5
    num_pulses = 300_000

    detector = _noiseless_detector()
    photon_numbers = rng.poisson(mu, size=num_pulses)

    observed = _signal_yield(detector, photon_numbers, rng)

    expected = 1.0 - math.exp(-mu)                       # standard Poisson: 0.3935
    buggy = 1.0 - math.exp(-mu * (1.0 - 1.0 / math.e))  # attenuated:       0.2711

    # Converges to the standard prediction (loose tolerance -- sampling noise).
    assert observed == pytest.approx(expected, abs=0.01), (
        f"yield {observed:.4f} != standard Poisson {expected:.4f}"
    )
    # And is inconsistent with the former double-thinning artifact.
    assert abs(observed - buggy) > 0.02, (
        f"yield {observed:.4f} is suspiciously close to the buggy attenuated "
        f"prediction {buggy:.4f}"
    )


def test_single_photon_per_pulse_always_clicks():
    """A pulse carrying exactly one photon at unity transmission/efficiency
    must click ~100% of the time.

    This is the sharpest exposure of the former bug: the old Bernoulli would
    have triggered on only ``1 - exp(-1) ~= 0.6321`` of such pulses, silently
    throwing away ~37% of single-photon events even with perfect transmission.
    """
    rng = np.random.default_rng(7)
    num_pulses = 100_000

    detector = _noiseless_detector()
    photon_numbers = np.ones(num_pulses, dtype=np.int64)

    observed = _signal_yield(detector, photon_numbers, rng)

    # Every pulse has one surviving photon -> every pulse must click.
    assert observed == pytest.approx(1.0, abs=0.005), (
        f"single-photon yield {observed:.4f} << 1.0; the detector is still "
        f"applying a redundant 1-exp(-1) attenuation"
    )
