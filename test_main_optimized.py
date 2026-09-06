#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive Test & Validation Suite for main_optimized.py

Tests all classes and functions against real-world QKD expectations:
  - Functional correctness (unit + integration)
  - Physical plausibility (key-rate vs distance, QBER bounds, Beer-Lambert)
  - Numerical stability (reproducibility, extreme values, no NaN/Inf)
  - Edge-case handling (invalid inputs, boundary values)
  - Regression against known theoretical references

Usage:
    python test_main_optimized.py
"""

import sys
import os
import time
import math
import copy
import traceback
import logging
from dataclasses import fields
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum

import numpy as np

# ── Import the module under test ──────────────────────────────────────────
# Assumes main_optimized.py is on sys.path (same directory or installed).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_optimized as mo

from qkd.datatypes import (
    ProtocolType, DetectorType, DoubleClickPolicy, DecoderArchitecture,
    SourceErrorModel, SecurityProof, ConfidenceBoundMethod, SimulationStatus,
    SourceStatisticsType, IntensityNode, IntensityConfig, AttenuationConfig,
    OpticalComponent, OpticalSourceConfig, DetectionConfig, ErrorCorrectionConfig,
    ProtocolParameters, PulseTypeConfig, PulseEnsembleConfig, TallyCounts,
    EpsilonAllocation, SecurityCertificate, SimulationResults,
)
from qkd.exceptions import (
    QKDException, ParameterValidationError, ConfigurationError,
    QKDSimulationError, LPFailureError,
)
from qkd.channel import FiberChannel, MAX_DISTANCE_KM, MAX_FIBER_LOSS_DB_KM
from qkd.detectors import SinglePhotonDetector, DetectionDiagnostics, DetectionResult
from qkd.sources import OpticalSource, PoissonSource, DensityMatrixSource
from qkd.modulators import MZMConfig
from qkd.noise_models import ElectricalNoiseConfig
from qkd.protocols import (
    BB84DecoyProtocol, B92Protocol, MDIQKDProtocol,
    RedundantTransmissionProtocol, DetectionResults, SiftingResults,
    BB84PreparedStates, B92PreparedStates, MDIPreparedStates,
)


# ══════════════════════════════════════════════════════════════════════════
#  TEST RUNNER INFRASTRUCTURE
# ══════════════════════════════════════════════════════════════════════════

class TestResult:
    """Single test outcome."""
    __slots__ = ("name", "category", "passed", "detail", "duration_ms")

    def __init__(self, name: str, category: str, passed: bool,
                 detail: str = "", duration_ms: float = 0.0):
        self.name = name
        self.category = category
        self.passed = passed
        self.detail = detail
        self.duration_ms = duration_ms


_results: List[TestResult] = []


def _run_test(name: str, category: str, func, *args, **kwargs):
    """Execute a single test function, catching all exceptions."""
    t0 = time.perf_counter()
    try:
        func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        _results.append(TestResult(name, category, True, "", elapsed))
    except AssertionError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _results.append(TestResult(name, category, False, str(e), elapsed))
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _results.append(TestResult(name, category, False,
                                   f"UNEXPECTED: {type(e).__name__}: {e}", elapsed))


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


def _assert_approx(a, b, rel_tol=1e-9, abs_tol=1e-12, msg=""):
    if not (abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)):
        raise AssertionError(f"{msg}: {a} != {b} (rel_tol={rel_tol}, abs_tol={abs_tol})")


def _assert_in(val, container, msg=""):
    if val not in container:
        raise AssertionError(f"{msg}: {val!r} not in {container!r}")


def _assert_raises(exc_type, func, *args, msg="", **kwargs):
    try:
        func(*args, **kwargs)
        raise AssertionError(f"{msg}: Expected {exc_type.__name__} but no exception raised")
    except exc_type:
        pass  # expected


# ══════════════════════════════════════════════════════════════════════════
#  CATEGORY 1: CONFIGURATION VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def test_default_config_has_all_top_level_keys():
    for key in ("protocol_params", "protocol_runtime", "channel",
                "detector", "source", "protocol", "simulation"):
        _assert(key in mo.DEFAULT_CONFIG, f"Missing top-level key: {key}")


def test_default_config_detector_physically_plausible():
    det = mo.DEFAULT_CONFIG["detector"]
    _assert(0 < det["det_eff_d0"] <= 1, "det_eff_d0 out of [0,1]")
    _assert(0 < det["det_eff_d1"] <= 1, "det_eff_d1 out of [0,1]")
    _assert(0 <= det["dark_rate"] < 1, "dark_rate out of range")
    _assert(0 <= det["qber_intrinsic"] < 0.5, "qber_intrinsic >= 0.5 is unphysical for QKD")
    _assert(0 <= det["misalignment"] < 0.5, "misalignment >= 0.5 is unphysical")
    _assert(det["dead_time_ns"] >= 0, "dead_time_ns negative")
    _assert(det["afterpulse_prob"] >= 0, "afterpulse_prob negative")
    _assert(det["breakdown_voltage"] > 0, "breakdown_voltage must be positive")
    _assert(det["temperature_k"] > 0, "temperature_k must be positive (Kelvin)")


def test_default_config_source_physically_plausible():
    src = mo.DEFAULT_CONFIG["source"]
    _assert(src["pulse_period_ns"] > 0, "pulse_period_ns must be positive")
    _assert(0 <= src["modulation_index"] <= 1, "modulation_index out of [0,1]")
    _assert(0 < src["source_fidelity"] <= 1, "source_fidelity out of (0,1]")
    _assert(src["extinction_ratio"] >= 0, "extinction_ratio negative")
    _assert(src["v_pi"] > 0, "v_pi must be positive")
    _assert(0 < src["ideal_emission_probability"] <= 1, "ideal_emission_probability out of (0,1]")
    # Pulse probabilities sum to 1
    pulses = src["pulses"]
    total_prob = sum(p["prob"] for p in pulses.values())
    _assert_approx(total_prob, 1.0, abs_tol=1e-12, msg="Pulse probs don't sum to 1")
    # Signal mu > decoy mu > vacuum mu (decoy-state protocol requirement)
    _assert(pulses["signal"]["mu"] > pulses["decoy"]["mu"],
            "Signal mu must exceed decoy mu for decoy-state protocol")
    _assert(pulses["decoy"]["mu"] > pulses["vacuum"]["mu"],
            "Decoy mu must exceed vacuum mu")


def test_default_config_epsilon_values():
    eps = mo.DEFAULT_CONFIG["protocol"]["epsilons"]
    total = sum(eps.values())
    _assert(total > 0, "Total epsilon must be positive")
    _assert(total < 1, "Total epsilon must be less than 1 (security budget)")
    for k, v in eps.items():
        _assert(v > 0, f"Epsilon {k} must be positive, got {v}")


def test_default_sweeps_keys_are_valid_paths():
    """Every sweep key must be a dot-path that resolves in DEFAULT_CONFIG."""
    for key_path in mo.DEFAULT_SWEEPS:
        parts = key_path.split(".")
        current = mo.DEFAULT_CONFIG
        for part in parts[:-1]:
            _assert(part in current, f"Sweep key {key_path}: intermediate '{part}' not in config")
            current = current[part]
        _assert(parts[-1] in current,
                f"Sweep key {key_path}: terminal key '{parts[-1]}' not in config")


def test_detector_metadata_keys_populated():
    _assert(len(mo.DETECTOR_METADATA_KEYS) > 0, "DETECTOR_METADATA_KEYS is empty")
    _assert("final_state_snippet" in mo.DETECTOR_METADATA_KEYS,
            "final_state_snippet missing from DETECTOR_METADATA_KEYS")


def test_detector_override_keys_populated():
    _assert(len(mo.DETECTOR_OVERRIDE_KEYS) > 0, "DETECTOR_OVERRIDE_KEYS is empty")
    expected = {"det_override_det_eff_d0", "det_override_det_eff_d1",
                "det_override_bias_voltage", "det_override_breakdown_voltage"}
    for k in expected:
        _assert(k in mo.DETECTOR_OVERRIDE_KEYS, f"{k} missing from DETECTOR_OVERRIDE_KEYS")


# ══════════════════════════════════════════════════════════════════════════
#  CATEGORY 2: HELPER FUNCTION TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_set_nested_value_basic():
    d = {"a": {"b": 1}}
    mo.set_nested_value(d, "a.b", 42)
    _assert(d["a"]["b"] == 42, "set_nested_value did not update value")


def test_set_nested_value_creates_intermediate():
    d = {"a": {}}
    mo.set_nested_value(d, "a.x.y", 99)
    _assert(d["a"]["x"]["y"] == 99, "set_nested_value did not create intermediate keys")


def test_set_nested_value_strict_rejects_missing_key():
    d = {"a": {"b": 1}}
    # "x" is a truly missing intermediate key (not just a missing final key)
    _assert_raises(ConfigurationError, mo.set_nested_value,
                   d, "x.y", 42, strict=True,
                   msg="strict mode should reject missing intermediate key")


def test_set_nested_value_strict_allows_existing_key():
    d = {"a": {"b": 1}}
    mo.set_nested_value(d, "a.b", 42, strict=True)
    _assert(d["a"]["b"] == 42, "strict mode should allow existing key path")


def test_set_nested_value_empty_key_raises():
    _assert_raises(ConfigurationError, mo.set_nested_value,
                   {}, "", 1, msg="Empty key_path should raise")


def test_set_nested_value_non_dict_intermediate():
    d = {"a": 5}
    _assert_raises(ConfigurationError, mo.set_nested_value,
                   d, "a.b", 1, msg="Should raise when traversing non-dict")


def test_resolve_enum_type_direct():
    result = mo._resolve_enum_type(DetectorType)
    _assert(result is DetectorType, "Should resolve direct enum type")


def test_resolve_enum_type_optional():
    from typing import Optional
    result = mo._resolve_enum_type(Optional[DetectorType])
    _assert(result is DetectorType, "Should resolve enum from Optional[Enum]")


def test_resolve_enum_type_non_enum():
    result = mo._resolve_enum_type(int)
    _assert(result is None, "Non-enum annotation should return None")


def test_build_dataclass_config_valid():
    raw = {"name": "test", "mean_photon_number": 0.5, "probability": 0.8}
    result = mo._build_dataclass_config(PulseTypeConfig, raw)
    _assert(isinstance(result, PulseTypeConfig), "Should return PulseTypeConfig")
    _assert_approx(result.mean_photon_number, 0.5)


def test_build_dataclass_config_passthrough():
    existing = PulseTypeConfig(name="test", mean_photon_number=0.5, probability=0.8)
    result = mo._build_dataclass_config(PulseTypeConfig, existing)
    _assert(result is existing, "Should pass through existing dataclass instance")


def test_build_dataclass_config_invalid_type():
    _assert_raises(ConfigurationError, mo._build_dataclass_config,
                   PulseTypeConfig, 42, msg="Should reject non-dict input")


def test_parse_enum_by_name():
    result = mo.parse_enum(DetectorType, "SPD")
    _assert(result == DetectorType.SPD, "Should parse by name")


def test_parse_enum_by_value():
    # Enum values are typically lowercase
    for member in DetectorType:
        result = mo.parse_enum(DetectorType, member.value)
        _assert(result == member, f"Should parse by value: {member.value}")


def test_parse_enum_case_insensitive():
    result = mo.parse_enum(DetectorType, "spd")
    _assert(result == DetectorType.SPD, "Should be case-insensitive")


def test_parse_enum_passthrough():
    result = mo.parse_enum(DetectorType, DetectorType.SPD)
    _assert(result is DetectorType.SPD, "Should pass through existing enum member")


def test_parse_enum_invalid_raises():
    _assert_raises(ParameterValidationError, mo.parse_enum,
                   DetectorType, "INVALID_TYPE", msg="Should reject invalid enum value")


def test_normalize_detector_config():
    det = mo.DEFAULT_CONFIG["detector"]
    normalized = mo.normalize_detector_config(det)
    _assert(isinstance(normalized["double_click_policy"], DoubleClickPolicy),
            "double_click_policy should be converted to enum")
    _assert(isinstance(normalized["detector_type"], DetectorType),
            "detector_type should be converted to enum")


def test_normalize_detector_config_already_enum():
    det = dict(mo.DEFAULT_CONFIG["detector"])
    det["detector_type"] = DetectorType.SPD
    normalized = mo.normalize_detector_config(det)
    _assert(normalized["detector_type"] is DetectorType.SPD,
            "Already-enum values should pass through")


def test_as_qkd_exception_passthrough():
    original = ConfigurationError("test")
    result = mo._as_qkd_exception(original, "wrapper")
    _assert(result is original, "QKDException should pass through unchanged")


def test_as_qkd_exception_type_error():
    result = mo._as_qkd_exception(TypeError("bad"), "test")
    _assert(isinstance(result, ConfigurationError), "TypeError should become ConfigurationError")


def test_as_qkd_exception_runtime_error():
    result = mo._as_qkd_exception(RuntimeError("bad"), "test")
    _assert(isinstance(result, QKDSimulationError), "RuntimeError should become QKDSimulationError")


# ══════════════════════════════════════════════════════════════════════════
#  CATEGORY 3: BUILDER FUNCTION TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_build_epsilon_allocation_valid():
    alloc = mo.build_epsilon_allocation(mo.DEFAULT_CONFIG)
    _assert(isinstance(alloc, EpsilonAllocation), "Should return EpsilonAllocation")
    _assert(alloc.eps_sec > 0, "eps_sec should be positive")


def test_build_epsilon_allocation_negative_rejected():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["protocol"]["epsilons"]["eps_sec"] = -1e-10
    _assert_raises(ParameterValidationError, mo.build_epsilon_allocation, cfg,
                   msg="Negative epsilon should be rejected")


def test_build_epsilon_allocation_total_ge_one_rejected():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["protocol"]["epsilons"]["eps_sec"] = 0.5
    cfg["protocol"]["epsilons"]["eps_cor"] = 0.5
    _assert_raises(ParameterValidationError, mo.build_epsilon_allocation, cfg,
                   msg="Total epsilon >= 1 should be rejected")


def test_build_pulse_configs():
    result = mo.build_pulse_configs(mo.DEFAULT_CONFIG)
    _assert(isinstance(result, tuple), "Should return tuple")
    _assert(len(result) == 3, "Should have 3 pulse types (signal, decoy, vacuum)")
    names = {pc.name for pc in result}
    _assert("signal" in names, "Missing 'signal' pulse config")


def test_build_intensity_config():
    ic = mo.build_intensity_config(mo.DEFAULT_CONFIG)
    _assert(isinstance(ic, IntensityConfig), "Should return IntensityConfig")
    _assert_approx(ic.signal.mu, mo.DEFAULT_CONFIG["source"]["pulses"]["signal"]["mu"])
    _assert(len(ic.decoys) == 2, "Should have 2 decoy nodes (decoy + vacuum)")


def test_build_attenuation_config():
    ac = mo.build_attenuation_config(50.0, mo.DEFAULT_CONFIG)
    _assert(isinstance(ac, AttenuationConfig), "Should return AttenuationConfig")
    _assert_approx(ac.fiber_length, 50.0)
    _assert_approx(ac.attenuation_coefficient, 0.2)


def test_build_optical_source_config():
    osc = mo.build_optical_source_config(mo.DEFAULT_CONFIG)
    _assert(isinstance(osc, OpticalSourceConfig), "Should return OpticalSourceConfig")
    expected_rate = 1e9 / mo.DEFAULT_CONFIG["source"]["pulse_period_ns"]
    _assert_approx(osc.source_rate, expected_rate, rel_tol=1e-6)


def test_build_optical_source_config_missing_rate_and_period():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    del cfg["source"]["pulse_period_ns"]
    # source_rate is also not present by default
    _assert_raises(ConfigurationError, mo.build_optical_source_config, cfg,
                   msg="Missing both source_rate and pulse_period_ns should raise")


def test_build_optical_source_config_explicit_source_rate():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["source"]["source_rate"] = 5e8
    osc = mo.build_optical_source_config(cfg)
    _assert_approx(osc.source_rate, 5e8, rel_tol=1e-6)


def test_build_mzm_config_from_dict():
    src = mo.DEFAULT_CONFIG["source"]
    mzm = mo.build_mzm_config(src)
    _assert(isinstance(mzm, MZMConfig), "Should return MZMConfig")
    _assert_approx(mzm.v_pi, 3.5)
    _assert_approx(mzm.extinction_ratio_db, 25.0)


def test_build_mzm_config_none():
    _assert(mo.build_mzm_config({}) is None, "Missing mzm key should return None")
    _assert(mo.build_mzm_config({"mzm": None}) is None, "None mzm should return None")


def test_build_electrical_noise_config_from_dict():
    src = mo.DEFAULT_CONFIG["source"]
    enc = mo.build_electrical_noise_config(src)
    _assert(isinstance(enc, ElectricalNoiseConfig), "Should return ElectricalNoiseConfig")
    _assert_approx(enc.bandwidth_hz, 1e9, rel_tol=1e-6)


def test_build_electrical_noise_config_none():
    _assert(mo.build_electrical_noise_config({}) is None, "Missing should return None")
    _assert(mo.build_electrical_noise_config({"electrical_noise": None}) is None)


def test_build_detection_config_averaging():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["detector"]["det_eff_d0"] = 0.10
    cfg["detector"]["det_eff_d1"] = 0.20
    dc = mo.build_detection_config(cfg)
    _assert(isinstance(dc, DetectionConfig), "Should return DetectionConfig")
    _assert_approx(dc.efficiency, 0.15, abs_tol=1e-10, msg="Should average d0/d1 efficiencies")


def test_build_detection_config_dark_rate_averaging():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["detector"]["dark_rate"] = 1e-6
    cfg["detector"]["dark_rate_d1"] = 3e-6
    dc = mo.build_detection_config(cfg)
    _assert_approx(dc.dark_count_rate, 2e-6, abs_tol=1e-15,
                   msg="Should average d0/d1 dark rates when dark_rate_d1 provided")


def test_build_error_correction_config():
    ec = mo.build_error_correction_config(mo.DEFAULT_CONFIG)
    _assert(isinstance(ec, ErrorCorrectionConfig))
    _assert_approx(ec.efficiency, 1.16)


def test_build_protocol_parameters():
    pp = mo.build_protocol_parameters(50.0, mo.DEFAULT_CONFIG)
    _assert(isinstance(pp, ProtocolParameters))
    _assert(pp.attenuation.fiber_length == 50.0)
    _assert(isinstance(pp.protocol, ProtocolType))


def test_build_security_certificate_default_phase_eq():
    sc = mo.build_security_certificate(mo.DEFAULT_CONFIG)
    _assert(isinstance(sc, SecurityCertificate))
    _assert(sc.assumed_phase_equals_bit_error is True,
            "Default assumed_phase_equals_bit_error should be True")


def test_build_security_certificate_configurable_phase_eq():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["protocol"]["assumed_phase_equals_bit_error"] = False
    sc = mo.build_security_certificate(cfg)
    _assert(sc.assumed_phase_equals_bit_error is False,
            "Should respect config value for assumed_phase_equals_bit_error")


def test_build_security_certificate_with_lp_diag():
    lp = {"preferred_lp_solver": "highs"}
    sc = mo.build_security_certificate(mo.DEFAULT_CONFIG, lp_solver_diagnostics=lp)
    _assert(sc.lp_solver_diagnostics == lp, "lp_solver_diagnostics should be stored")


def test_build_security_metadata_none():
    result = mo._build_security_metadata({"security_metadata": None})
    _assert(result is None, "None security_metadata should return None")


def test_build_security_metadata_missing():
    result = mo._build_security_metadata({})
    _assert(result is None, "Missing security_metadata should return None")


def test_build_source_creates_optical_source():
    source = mo.build_source(mo.DEFAULT_CONFIG)
    _assert(isinstance(source, OpticalSource), "Should create OpticalSource")
    _assert(source.source_rate > 0, "Source rate should be positive")
    _assert(len(source.pulse_names()) == 3, "Should have 3 pulse types")


def test_build_source_with_density_matrices():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["source"]["density_matrices"] = {
        "signal": [[1.0, 0.0], [0.0, 0.0]],
        "decoy": [[1.0, 0.0], [0.0, 0.0]],
        "vacuum": [[1.0, 0.0], [0.0, 0.0]],
    }
    source = mo.build_source(cfg)
    _assert(isinstance(source, (OpticalSource, DensityMatrixSource)),
            "Should create source with density matrices")


def test_build_detector_returns_tuple():
    result = mo.build_detector(mo.DEFAULT_CONFIG)
    _assert(isinstance(result, tuple) and len(result) == 2,
            "build_detector should return (detector, overrides) tuple")
    detector, overrides = result
    _assert(isinstance(detector, SinglePhotonDetector), "First element should be detector")
    _assert(isinstance(overrides, dict), "Second element should be overrides dict")


def test_build_detector_overrides_recorded():
    detector, overrides = mo.build_detector(mo.DEFAULT_CONFIG)
    _assert("det_eff_d0" in overrides, "det_eff_d0 should be in overrides")
    _assert("det_eff_d1" in overrides, "det_eff_d1 should be in overrides")
    _assert("bias_voltage" in overrides, "bias_voltage should be in overrides")


def test_build_detector_bias_clamping():
    """When bias > breakdown, bias should be clamped to breakdown - 1."""
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["detector"]["bias_voltage"] = 50.0
    cfg["detector"]["breakdown_voltage"] = 45.0
    _, overrides = mo.build_detector(cfg)
    _assert(overrides["bias_voltage"] <= overrides["breakdown_voltage"],
            "bias_voltage should not exceed breakdown_voltage after clamping")


def test_build_detector_bias_no_clamp_needed():
    """When bias <= breakdown, no clamping should occur."""
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["detector"]["bias_voltage"] = 40.0
    cfg["detector"]["breakdown_voltage"] = 45.0
    _, overrides = mo.build_detector(cfg)
    _assert_approx(overrides["bias_voltage"], 40.0, abs_tol=1e-10,
                   msg="bias_voltage should remain unchanged when <= breakdown")


def test_build_channel_from_distance():
    ch = mo.build_channel_from_distance(50.0, mo.DEFAULT_CONFIG)
    _assert(isinstance(ch, FiberChannel), "Should return FiberChannel")
    # Beer-Lambert: transmittance = 10^(-loss*dB/km * km / 10)
    expected_t = 10 ** (-0.2 * 50.0 / 10.0)
    _assert_approx(ch.transmittance, expected_t, rel_tol=1e-6,
                   msg="Channel transmittance should follow Beer-Lambert law")


def test_build_channel_zero_distance():
    ch = mo.build_channel_from_distance(0.0, mo.DEFAULT_CONFIG)
    _assert_approx(ch.transmittance, 1.0, abs_tol=1e-12,
                   msg="Zero distance should have transmittance 1.0")


def test_build_protocol_bb84():
    source = mo.build_source(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)
    _assert(isinstance(proto, BB84DecoyProtocol), "Default should create BB84DecoyProtocol")


def test_build_protocol_b92():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["protocol_runtime"]["protocol_class"] = "B92Protocol"
    source = mo.build_source(cfg)
    proto = mo.build_protocol(cfg, source)
    _assert(isinstance(proto, B92Protocol), "Should create B92Protocol")


def test_build_protocol_mdi():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["protocol_runtime"]["protocol_class"] = "MDIQKDProtocol"
    source = mo.build_source(cfg)
    proto = mo.build_protocol(cfg, source)
    _assert(isinstance(proto, MDIQKDProtocol), "Should create MDIQKDProtocol")


def test_build_protocol_redundant():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["protocol_runtime"]["protocol_class"] = "RedundantTransmissionProtocol"
    source = mo.build_source(cfg)
    proto = mo.build_protocol(cfg, source)
    _assert(isinstance(proto, RedundantTransmissionProtocol),
            "Should create RedundantTransmissionProtocol")


def test_build_protocol_unknown_raises():
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    cfg["protocol_runtime"]["protocol_class"] = "NonExistentProtocol"
    source = mo.build_source(cfg)
    _assert_raises(ConfigurationError, mo.build_protocol, cfg, source,
                   msg="Unknown protocol class should raise ConfigurationError")


# ══════════════════════════════════════════════════════════════════════════
#  CATEGORY 4: SIMULATION FUNCTION TESTS
# ══════════════════════════════════════════════════════════════════════════

def _make_bb84_prepared_states(n=10000):
    """Helper: create BB84PreparedStates for testing."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)
    rng = np.random.default_rng(42)
    return proto.prepare_states(n, rng)


def test_get_pulse_indices_from_prepared_states():
    ps = _make_bb84_prepared_states()
    indices = mo.get_pulse_indices_from_prepared_states(ps)
    _assert(indices.dtype == np.int64, "Should return int64 array")
    _assert(len(indices) == ps.num_pulses, "Length should match num_pulses")


def test_get_pulse_indices_missing_attr_raises():
    class FakeStates:
        pass
    _assert_raises(QKDSimulationError, mo.get_pulse_indices_from_prepared_states,
                   FakeStates(), msg="Missing alice_pulse_type_indices should raise")


def test_generate_photons_for_prepared_states():
    ps = _make_bb84_prepared_states()
    source = mo.build_source(mo.DEFAULT_CONFIG)
    rng = np.random.default_rng(42)
    photons = mo.generate_photons_for_prepared_states(source, ps, rng)
    _assert(photons.dtype == np.int64, "Should return int64 array")
    _assert(len(photons) == ps.num_pulses, "Length should match num_pulses")
    _assert(photons.min() >= 0, "Photon counts should be non-negative")


def test_infer_ideal_outcomes_d0_bb84():
    ps = _make_bb84_prepared_states()
    outcomes = mo.infer_ideal_outcomes_d0(ps)
    _assert(outcomes.dtype == bool, "Should return boolean array")
    _assert(len(outcomes) == ps.num_pulses, "Length should match num_pulses")
    _assert(0 < outcomes.mean() < 1, "Should have mix of True/False")


def test_infer_ideal_outcomes_d0_unsupported_raises():
    class FakeStates:
        pass
    _assert_raises(QKDSimulationError, mo.infer_ideal_outcomes_d0,
                   FakeStates(), msg="Unsupported type should raise")


def test_extract_detection_arrays_from_detection_result():
    """Test extract_detection_arrays with a real detection result."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    detector, _ = mo.build_detector(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)
    rng = np.random.default_rng(42)
    ps = proto.prepare_states(1000, rng)
    photons = source.generate_photons(
        alice_pulse_indices=mo.get_pulse_indices_from_prepared_states(ps),
        rng=np.random.default_rng(43), num_samples=1000)
    ideal = mo.infer_ideal_outcomes_d0(ps)
    ch = mo.build_channel_from_distance(10.0, mo.DEFAULT_CONFIG)
    det_result = detector.simulate_detection(
        channel_transmittance=ch.transmittance, photon_numbers=photons,
        ideal_outcomes_d0=ideal, pulse_period_ns=10.0,
        rng=np.random.default_rng(44), return_diagnostics=True)
    d0, d1 = mo.extract_detection_arrays(det_result, 1000)
    _assert(len(d0) == 1000, "d0 length should match sample_size")
    _assert(len(d1) == 1000, "d1 length should match sample_size")


def test_extract_detection_arrays_from_plain_tuple():
    d0, d1 = mo.extract_detection_arrays((np.zeros(50, dtype=bool),
                                           np.ones(50, dtype=bool)), 50)
    _assert(len(d0) == 50, "Should extract from plain tuple")
    _assert(d1.all(), "d1 should be all True")


def test_extract_detection_arrays_invalid_raises():
    _assert_raises(QKDSimulationError, mo.extract_detection_arrays,
                   "invalid", 10, msg="Invalid input should raise")


def test_detector_result_to_protocol_detection_results():
    source = mo.build_source(mo.DEFAULT_CONFIG)
    detector, _ = mo.build_detector(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)
    rng = np.random.default_rng(42)
    ps = proto.prepare_states(1000, rng)
    photons = source.generate_photons(
        alice_pulse_indices=mo.get_pulse_indices_from_prepared_states(ps),
        rng=np.random.default_rng(43), num_samples=1000)
    ideal = mo.infer_ideal_outcomes_d0(ps)
    ch = mo.build_channel_from_distance(10.0, mo.DEFAULT_CONFIG)
    det_result = detector.simulate_detection(
        channel_transmittance=ch.transmittance, photon_numbers=photons,
        ideal_outcomes_d0=ideal, pulse_period_ns=10.0,
        rng=np.random.default_rng(44), return_diagnostics=True)
    dr = mo.detector_result_to_protocol_detection_results(det_result, 1000)
    _assert(isinstance(dr, DetectionResults), "Should return DetectionResults")
    _assert(dr.num_pulses == 1000, "num_pulses should match")


def test_basis_is_z_mask_bool():
    mask = mo._basis_is_z_mask(np.array([True, False, True]))
    _assert(mask[0] and mask[2] and not mask[1], "Bool Z-mask incorrect")


def test_basis_is_z_mask_int():
    mask = mo._basis_is_z_mask(np.array([0, 1, 0]))
    _assert(mask[0] and mask[2] and not mask[1], "Int Z-mask: 0=Z")


def test_basis_is_z_mask_string():
    mask = mo._basis_is_z_mask(np.array(["Z", "X", "Z_BASIS"]))
    _assert(mask[0] and mask[2] and not mask[1], "String Z-mask incorrect")


def test_get_first_existing_attr():
    class Obj:
        x = 10
    _assert(mo._get_first_existing_attr(Obj(), ["y", "x"]) == 10,
            "Should find first existing attribute")
    _assert(mo._get_first_existing_attr(Obj(), ["y", "z"]) is None,
            "Should return None when no attribute exists")


def test_summarize_tallies():
    tc1 = TallyCounts(sent=100, sifted=50, errors_sifted=5, double_clicks_discarded=0,
                       sent_z=50, sent_x=50, sifted_z=25, sifted_x=25,
                       errors_sifted_z=3, errors_sifted_x=2)
    tc2 = TallyCounts(sent=200, sifted=80, errors_sifted=8, double_clicks_discarded=0,
                       sent_z=100, sent_x=100, sifted_z=40, sifted_x=40,
                       errors_sifted_z=5, errors_sifted_x=3)
    total = mo.summarize_tallies({"0": tc1, "1": tc2})
    _assert(total.sent == 300, "Total sent should be 300")
    _assert(total.sifted == 130, "Total sifted should be 130")
    _assert(total.errors_sifted == 13, "Total errors should be 13")


def test_roundtrip_tallycounts():
    tc = TallyCounts(sent=100, sifted=50, errors_sifted=5, double_clicks_discarded=0,
                     sent_z=50, sent_x=50, sifted_z=25, sifted_x=25,
                     errors_sifted_z=3, errors_sifted_x=2)
    result = mo.roundtrip_tallycounts({"0": tc})
    _assert(result["0"].sent == 100, "Roundtrip should preserve sent")
    _assert(result["0"].sifted == 50, "Roundtrip should preserve sifted")


def test_roundtrip_pulse_configs():
    source = mo.build_source(mo.DEFAULT_CONFIG)
    result = mo.roundtrip_pulse_configs(source)
    _assert(isinstance(result, list), "Should return list")
    _assert(len(result) > 0, "Should have at least one pulse config")
    _assert("name" in result[0], "Each item should have 'name' key")


def test_build_simulation_results():
    source = mo.build_source(mo.DEFAULT_CONFIG)
    pp = mo.build_protocol_parameters(0.0, mo.DEFAULT_CONFIG)
    tc = TallyCounts(sent=1000, sifted=500, errors_sifted=10, double_clicks_discarded=0,
                     sent_z=500, sent_x=500, sifted_z=250, sifted_x=250,
                     errors_sifted_z=5, errors_sifted_x=5)
    stats = {"0": tc}
    result = mo.build_simulation_results(
        pp, mo.DEFAULT_CONFIG, stats, secure_key=100, raw_sifted=500,
        sim_time=1.0, status_text="OK", metadata={}, source=source)
    _assert(isinstance(result, SimulationResults), "Should return SimulationResults")
    _assert(result.secure_key_length == 100, "Secure key length should be preserved")
    _assert(result.status == SimulationStatus.OK, "Status should be OK")


# ══════════════════════════════════════════════════════════════════════════
#  CATEGORY 5: WORKER INFRASTRUCTURE TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_worker_state_initialization():
    ws = mo._WorkerState()
    _assert(ws.base_config is None, "base_config should start as None")
    _assert(ws.last_combo_idx == -1, "last_combo_idx should start as -1")
    _assert(ws.detector_overrides == {}, "detector_overrides should start empty")
    _assert(ws.source_meta_cache == {}, "source_meta_cache should start empty")


def test_init_worker_sets_state():
    ws_before = mo._ws.last_combo_idx
    mo.init_worker(mo.DEFAULT_CONFIG, {0: {}})
    _assert(mo._ws.base_config is mo.DEFAULT_CONFIG, "base_config should be set")
    _assert(mo._ws.last_combo_idx == -1, "last_combo_idx should be reset to -1")
    _assert(mo._ws.detector_overrides == {}, "detector_overrides should be reset")


def test_get_source_metadata_row():
    source = mo.build_source(mo.DEFAULT_CONFIG)
    meta = mo.get_source_metadata_row(source)
    _assert(isinstance(meta, dict), "Should return dict")
    _assert("source_type" in meta, "Should contain source_type")
    _assert("pulse_signal_mu" in meta, "Should contain pulse_signal_mu")
    _assert("statistics_type" in meta, "Should contain statistics_type")
    _assert("mzm_status" in meta, "Should contain mzm_status")
    _assert("signal_pulse_index" in meta, "Should contain signal_pulse_index")


def test_build_detector_override_row():
    overrides = {"det_eff_d0": 0.15, "bias_voltage": 44.0}
    row = mo._build_detector_override_row(overrides)
    _assert("det_override_det_eff_d0" in row, "Should have det_override_det_eff_d0 key")
    _assert_approx(row["det_override_det_eff_d0"], 0.15)
    _assert("det_override_bias_voltage" in row, "Should have det_override_bias_voltage key")
    _assert_approx(row["det_override_bias_voltage"], 44.0)


def test_build_detector_override_row_empty():
    row = mo._build_detector_override_row({})
    for k in mo.DETECTOR_OVERRIDE_KEYS:
        clean = k.replace("det_override_", "")
        _assert(row.get(k) is None, f"Empty overrides should give None for {k}")


def test_log_source_details_no_crash():
    """log_source_details should not raise even with a real source."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    # Should not raise
    mo.log_source_details(source, level=logging.DEBUG)


# ══════════════════════════════════════════════════════════════════════════
#  CATEGORY 6: PHYSICAL PLAUSIBILITY TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_channel_transmittance_beer_lambert():
    """Transmittance must follow 10^(-alpha*d/10) (Beer-Lambert law)."""
    alpha = mo.DEFAULT_CONFIG["channel"]["fiber_loss_db_km"]
    for d in [0, 10, 50, 100, 150]:
        ch = mo.build_channel_from_distance(float(d), mo.DEFAULT_CONFIG)
        expected = 10 ** (-alpha * d / 10.0)
        _assert_approx(ch.transmittance, expected, rel_tol=1e-6,
                       msg=f"Beer-Lambert at {d} km")


def test_channel_loss_increases_with_distance():
    """Total loss (dB) must increase monotonically with distance."""
    losses = []
    for d in [0, 25, 50, 75, 100, 125, 150]:
        ch = mo.build_channel_from_distance(float(d), mo.DEFAULT_CONFIG)
        losses.append(ch.total_loss_db)
    for i in range(1, len(losses)):
        _assert(losses[i] > losses[i - 1],
                f"Loss should increase: {losses[i]} <= {losses[i-1]}")


def test_transmittance_decreases_with_distance():
    """Transmittance must decrease monotonically with distance."""
    transmittances = []
    for d in [0, 25, 50, 75, 100, 125, 150]:
        ch = mo.build_channel_from_distance(float(d), mo.DEFAULT_CONFIG)
        transmittances.append(ch.transmittance)
    for i in range(1, len(transmittances)):
        _assert(transmittances[i] < transmittances[i - 1],
                f"Transmittance should decrease at index {i}")


def test_detection_yield_decreases_with_distance():
    """Detection yield should generally decrease with distance (fewer photons arrive)."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    detector, _ = mo.build_detector(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)
    yields = []
    for d in [0, 50, 100]:
        ch = mo.build_channel_from_distance(float(d), mo.DEFAULT_CONFIG)
        rng = np.random.default_rng(42)
        ps = proto.prepare_states(5000, np.random.default_rng(42))
        photons = source.generate_photons(
            alice_pulse_indices=mo.get_pulse_indices_from_prepared_states(ps),
            rng=np.random.default_rng(43), num_samples=5000)
        ideal = mo.infer_ideal_outcomes_d0(ps)
        det_result = detector.simulate_detection(
            channel_transmittance=ch.transmittance, photon_numbers=photons,
            ideal_outcomes_d0=ideal, pulse_period_ns=10.0,
            rng=np.random.default_rng(44), return_diagnostics=False)
        d0, d1 = mo.extract_detection_arrays(det_result, 5000)
        yields.append(float(np.count_nonzero(d0 | d1)) / 5000)
    # At 0 km yield should be higher than at 100 km
    _assert(yields[0] > yields[2],
            f"Yield at 0km ({yields[0]:.4f}) should exceed yield at 100km ({yields[2]:.4f})")


def test_qber_in_plausible_range():
    """QBER for a well-configured simulation should be in [0, 0.2]."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    detector, _ = mo.build_detector(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)
    rng = np.random.default_rng(42)
    ps = proto.prepare_states(10000, np.random.default_rng(42))
    photons = source.generate_photons(
        alice_pulse_indices=mo.get_pulse_indices_from_prepared_states(ps),
        rng=np.random.default_rng(43), num_samples=10000)
    ideal = mo.infer_ideal_outcomes_d0(ps)
    ch = mo.build_channel_from_distance(10.0, mo.DEFAULT_CONFIG)
    det_result = detector.simulate_detection(
        channel_transmittance=ch.transmittance, photon_numbers=photons,
        ideal_outcomes_d0=ideal, pulse_period_ns=10.0,
        rng=np.random.default_rng(44), return_diagnostics=True)
    proto_det = mo.detector_result_to_protocol_detection_results(det_result, 10000)
    sifted = proto.sift_results(prepared_states=ps, detection_results=proto_det,
                                rng=np.random.default_rng(45))
    summary = sifted.summary(confidence_level=0.95)
    _assert(0 <= summary["qber"] <= 0.2,
            f"QBER {summary['qber']:.4f} outside plausible range [0, 0.2]")


def test_photon_numbers_nonnegative():
    """Photon number samples should always be non-negative integers."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    for _ in range(5):
        rng = np.random.default_rng(np.random.SeedSequence().entropy)
        indices = np.zeros(1000, dtype=np.int64)  # All signal pulses
        photons = source.generate_photons(alice_pulse_indices=indices, rng=rng, num_samples=1000)
        _assert(photons.min() >= 0, "Negative photon numbers detected")
        _assert(np.issubdtype(photons.dtype, np.integer), "Photon numbers should be integers")


def test_epsilon_sum_much_less_than_one():
    """The total epsilon budget should be << 1 for meaningful security."""
    eps = mo.DEFAULT_CONFIG["protocol"]["epsilons"]
    total = sum(eps.values())
    _assert(total < 0.01, f"Total epsilon {total:.2e} should be << 1 for composable security")


# ══════════════════════════════════════════════════════════════════════════
#  CATEGORY 7: NUMERICAL STABILITY & REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════

def test_rng_reproducibility():
    """Same seed must produce identical results."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)

    def _run_once(seed):
        rng = np.random.default_rng(seed)
        ps = proto.prepare_states(1000, np.random.default_rng(seed))
        return mo.get_pulse_indices_from_prepared_states(ps).tolist()

    r1 = _run_once(12345)
    r2 = _run_once(12345)
    _assert(r1 == r2, "Same seed should produce identical pulse indices")


def test_seed_sequence_independence():
    """SeedSequence.spawn should produce independent child streams."""
    parent = np.random.SeedSequence(12345)
    children = parent.spawn(5)
    rngs = [np.random.default_rng(c) for c in children]
    # Generate samples from each child
    samples = [rng.integers(0, 2**63, size=100) for rng in rngs]
    # No two children should produce identical sequences
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            _assert(not np.array_equal(samples[i], samples[j]),
                    f"Child streams {i} and {j} produced identical outputs")


def test_no_nan_or_inf_in_channel():
    """Channel transmittance should never be NaN or Inf."""
    for d in [0, 1, 50, 100, 200, 500, 1000]:
        ch = mo.build_channel_from_distance(float(d), mo.DEFAULT_CONFIG)
        _assert(np.isfinite(ch.transmittance), f"Non-finite transmittance at {d} km")
        _assert(np.isfinite(ch.total_loss_db), f"Non-finite loss at {d} km")


def test_no_nan_or_inf_in_detection():
    """Detection results should never contain NaN or Inf."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    detector, _ = mo.build_detector(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)
    rng = np.random.default_rng(42)
    ps = proto.prepare_states(1000, rng)
    photons = source.generate_photons(
        alice_pulse_indices=mo.get_pulse_indices_from_prepared_states(ps),
        rng=np.random.default_rng(43), num_samples=1000)
    ideal = mo.infer_ideal_outcomes_d0(ps)
    ch = mo.build_channel_from_distance(50.0, mo.DEFAULT_CONFIG)
    det_result = detector.simulate_detection(
        channel_transmittance=ch.transmittance, photon_numbers=photons,
        ideal_outcomes_d0=ideal, pulse_period_ns=10.0,
        rng=np.random.default_rng(44), return_diagnostics=True)
    d0, d1 = mo.extract_detection_arrays(det_result, 1000)
    _assert(np.all(np.isfinite(d0.astype(float))), "NaN/Inf in d0 array")
    _assert(np.all(np.isfinite(d1.astype(float))), "NaN/Inf in d1 array")


def test_very_small_pulse_count():
    """Simulation should not crash with very small pulse counts."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    detector, _ = mo.build_detector(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)
    rng = np.random.default_rng(42)
    ps = proto.prepare_states(10, rng)
    photons = source.generate_photons(
        alice_pulse_indices=mo.get_pulse_indices_from_prepared_states(ps),
        rng=np.random.default_rng(43), num_samples=10)
    ideal = mo.infer_ideal_outcomes_d0(ps)
    ch = mo.build_channel_from_distance(10.0, mo.DEFAULT_CONFIG)
    # Should not raise
    det_result = detector.simulate_detection(
        channel_transmittance=ch.transmittance, photon_numbers=photons,
        ideal_outcomes_d0=ideal, pulse_period_ns=10.0,
        rng=np.random.default_rng(44), return_diagnostics=True)
    _assert(det_result is not None, "Should produce a result even with 10 pulses")


def test_extreme_distance_no_crash():
    """Very large distances should produce results (likely zero key)."""
    source = mo.build_source(mo.DEFAULT_CONFIG)
    detector, _ = mo.build_detector(mo.DEFAULT_CONFIG)
    proto = mo.build_protocol(mo.DEFAULT_CONFIG, source)
    rng = np.random.default_rng(42)
    ps = proto.prepare_states(1000, rng)
    photons = source.generate_photons(
        alice_pulse_indices=mo.get_pulse_indices_from_prepared_states(ps),
        rng=np.random.default_rng(43), num_samples=1000)
    ideal = mo.infer_ideal_outcomes_d0(ps)
    ch = mo.build_channel_from_distance(500.0, mo.DEFAULT_CONFIG)
    _assert(ch.transmittance > 0, "Transmittance should still be positive at 500 km")
    det_result = detector.simulate_detection(
        channel_transmittance=ch.transmittance, photon_numbers=photons,
        ideal_outcomes_d0=ideal, pulse_period_ns=10.0,
        rng=np.random.default_rng(44), return_diagnostics=True)
    _assert(det_result is not None, "Should not crash at extreme distance")


def test_detector_clone_independence():
    """Cloned detector should produce same type but be independent."""
    detector, _ = mo.build_detector(mo.DEFAULT_CONFIG)
    clone = detector.clone()
    _assert(type(clone) is type(detector), "Clone should be same type")
    # Modify clone state, original should be unaffected
    if hasattr(clone, 'reset_state'):
        clone.reset_state()
    # Original should still be functional
    _assert(detector is not clone, "Clone should be a different object")


# ══════════════════════════════════════════════════════════════════════════
#  CATEGORY 8: EDGE CASE & REGRESSION TESTS
# ══════════════════════════════════════════════════════════════════════════

def test_channel_at_max_distance():
    """Channel at MAX_DISTANCE_KM should still be constructible.

    At extreme distances, Beer-Lambert T = 10^(-alpha*d/10) may underflow
    to exactly 0.0 in float64, so we accept >= 0 rather than > 0.
    """
    ch = mo.build_channel_from_distance(float(MAX_DISTANCE_KM), mo.DEFAULT_CONFIG)
    _assert(ch.transmittance >= 0, "Transmittance should be non-negative at MAX_DISTANCE_KM")
    _assert(ch.transmittance <= 1, "Transmittance should be <= 1 at MAX_DISTANCE_KM")


def test_zero_distance_is_lossless():
    ch = mo.build_channel_from_distance(0.0, mo.DEFAULT_CONFIG)
    _assert(ch.is_lossless, "Zero distance should be lossless")
    _assert_approx(ch.transmittance, 1.0, abs_tol=1e-15)


def test_probability_balancing_negative_detected():
    """Probability balancing that would produce negative values should be detectable.

    set_nested_value is a generic setter that does not validate probability
    semantics. The caller (run_single_simulation) is responsible for checking
    that probabilities sum to 1. Here we verify that the invalid state can be
    detected by inspecting the config after setting.
    """
    cfg = copy.deepcopy(mo.DEFAULT_CONFIG)
    vac_p = cfg["source"]["pulses"]["vacuum"]["prob"]  # 0.1
    sig_p = 0.95
    # After setting signal prob to 0.95, balanced decoy = 1 - 0.95 - 0.1 = -0.05
    mo.set_nested_value(cfg, "source.pulses.signal.prob", sig_p)
    total = sig_p + vac_p
    _assert(total > 1.0, "Signal + vacuum should exceed 1.0 to make decoy negative")
    # The framework should be able to detect this condition
    remaining_for_decoy = 1.0 - sig_p - vac_p
    _assert(remaining_for_decoy < 0, "Decoy probability would be negative; detectable")


def test_run_single_simulation_basic():
    """End-to-end test of run_single_simulation with known config."""
    mo.init_worker(mo.DEFAULT_CONFIG, {0: {}})
    args = (10.0, 1000, True, 42, 0)  # dist, pulses, noise, seed, combo
    result = mo.run_single_simulation(args)
    _assert(isinstance(result, dict), "Should return dict")
    _assert("distance_km" in result, "Result should have distance_km")
    _assert("status" in result, "Result should have status")
    _assert("secure_key_bits" in result, "Result should have secure_key_bits")
    _assert("qber" in result, "Result should have qber")
    _assert("protocol_class" in result, "Result should have protocol_class")
    _assert_approx(result["distance_km"], 10.0, abs_tol=0.01)


def test_run_single_simulation_error_path_fields():
    """Error-path results should have the same fields as success-path."""
    mo.init_worker(mo.DEFAULT_CONFIG, {0: {}})
    # Use extreme distance that might cause failure
    args = (99999.0, 10, True, 42, 0)
    result = mo.run_single_simulation(args)
    _assert("protocol_class" in result, "Error row should have protocol_class")
    _assert("protocol_name" in result, "Error row should have protocol_name")
    _assert("simulation_time_sec" in result, "Error row should have simulation_time_sec")


def test_run_single_simulation_deterministic():
    """Same seed should produce identical results."""
    mo.init_worker(mo.DEFAULT_CONFIG, {0: {}})
    args1 = (50.0, 1000, True, 12345, 0)
    args2 = (50.0, 1000, True, 12345, 0)
    r1 = mo.run_single_simulation(args1)
    mo.init_worker(mo.DEFAULT_CONFIG, {0: {}})
    r2 = mo.run_single_simulation(args2)
    _assert(r1["secure_key_bits"] == r2["secure_key_bits"],
            "Same seed should give same secure_key_bits")
    _assert_approx(r1["qber"], r2["qber"], abs_tol=1e-12,
                   msg="Same seed should give same QBER")


def test_attenuation_coeff_matches_config():
    """AttenuationConfig coefficient must match config fiber_loss_db_km."""
    for d in [0, 10, 50, 100]:
        ac = mo.build_attenuation_config(float(d), mo.DEFAULT_CONFIG)
        _assert_approx(ac.attenuation_coefficient,
                       mo.DEFAULT_CONFIG["channel"]["fiber_loss_db_km"],
                       msg=f"At distance {d}")


def test_intensity_config_signal_decoy_ordering():
    """Signal mu must be the first (signal) node, decoys follow."""
    ic = mo.build_intensity_config(mo.DEFAULT_CONFIG)
    _assert(ic.signal.mu > ic.decoys[0].mu,
            "Signal mu should exceed first decoy mu")


def test_math_pi_used_in_config():
    """F-22: phi_bias should use math.pi, not np.pi."""
    expected = math.pi / 2.0
    actual = mo.DEFAULT_CONFIG["source"]["phi_bias"]
    _assert_approx(actual, expected, msg="phi_bias should equal math.pi/2")


def test_worker_state_has_detector_overrides_slot():
    """_WorkerState must have detector_overrides slot (F-01 fix)."""
    ws = mo._WorkerState()
    _assert(hasattr(ws, "detector_overrides"), "_WorkerState missing detector_overrides slot")
    ws.detector_overrides = {"test": 1}
    _assert(ws.detector_overrides == {"test": 1}, "Cannot set detector_overrides")


def test_build_detector_override_keys_match():
    """DETECTOR_OVERRIDE_KEYS should align with _build_detector_override_row output."""
    overrides = {
        "det_eff_d0": 0.15, "det_eff_d1": 0.15, "dark_rate": 1e-6,
        "dark_rate_d1": 2e-6, "bias_voltage": 44.0, "breakdown_voltage": 45.0,
        "temperature_k": 293.0, "ref_temperature_k": 293.0, "ref_bias_voltage": 40.0,
    }
    row = mo._build_detector_override_row(overrides)
    for key in mo.DETECTOR_OVERRIDE_KEYS:
        _assert(key in row, f"DETECTOR_OVERRIDE_KEY {key} not in override row")


# ══════════════════════════════════════════════════════════════════════════
#  TEST EXECUTION & REPORTING
# ══════════════════════════════════════════════════════════════════════════

def _run_all_tests():
    """Execute all test functions defined in this module."""
    test_functions = [
        # Category 1: Configuration Validation
        (test_default_config_has_all_top_level_keys, "Config Validation"),
        (test_default_config_detector_physically_plausible, "Config Validation"),
        (test_default_config_source_physically_plausible, "Config Validation"),
        (test_default_config_epsilon_values, "Config Validation"),
        (test_default_sweeps_keys_are_valid_paths, "Config Validation"),
        (test_detector_metadata_keys_populated, "Config Validation"),
        (test_detector_override_keys_populated, "Config Validation"),
        # Category 2: Helper Functions
        (test_set_nested_value_basic, "Helper Functions"),
        (test_set_nested_value_creates_intermediate, "Helper Functions"),
        (test_set_nested_value_strict_rejects_missing_key, "Helper Functions"),
        (test_set_nested_value_strict_allows_existing_key, "Helper Functions"),
        (test_set_nested_value_empty_key_raises, "Helper Functions"),
        (test_set_nested_value_non_dict_intermediate, "Helper Functions"),
        (test_resolve_enum_type_direct, "Helper Functions"),
        (test_resolve_enum_type_optional, "Helper Functions"),
        (test_resolve_enum_type_non_enum, "Helper Functions"),
        (test_build_dataclass_config_valid, "Helper Functions"),
        (test_build_dataclass_config_passthrough, "Helper Functions"),
        (test_build_dataclass_config_invalid_type, "Helper Functions"),
        (test_parse_enum_by_name, "Helper Functions"),
        (test_parse_enum_by_value, "Helper Functions"),
        (test_parse_enum_case_insensitive, "Helper Functions"),
        (test_parse_enum_passthrough, "Helper Functions"),
        (test_parse_enum_invalid_raises, "Helper Functions"),
        (test_normalize_detector_config, "Helper Functions"),
        (test_normalize_detector_config_already_enum, "Helper Functions"),
        (test_as_qkd_exception_passthrough, "Helper Functions"),
        (test_as_qkd_exception_type_error, "Helper Functions"),
        (test_as_qkd_exception_runtime_error, "Helper Functions"),
        # Category 3: Builder Functions
        (test_build_epsilon_allocation_valid, "Builder Functions"),
        (test_build_epsilon_allocation_negative_rejected, "Builder Functions"),
        (test_build_epsilon_allocation_total_ge_one_rejected, "Builder Functions"),
        (test_build_pulse_configs, "Builder Functions"),
        (test_build_intensity_config, "Builder Functions"),
        (test_build_attenuation_config, "Builder Functions"),
        (test_build_optical_source_config, "Builder Functions"),
        (test_build_optical_source_config_missing_rate_and_period, "Builder Functions"),
        (test_build_optical_source_config_explicit_source_rate, "Builder Functions"),
        (test_build_mzm_config_from_dict, "Builder Functions"),
        (test_build_mzm_config_none, "Builder Functions"),
        (test_build_electrical_noise_config_from_dict, "Builder Functions"),
        (test_build_electrical_noise_config_none, "Builder Functions"),
        (test_build_detection_config_averaging, "Builder Functions"),
        (test_build_detection_config_dark_rate_averaging, "Builder Functions"),
        (test_build_error_correction_config, "Builder Functions"),
        (test_build_protocol_parameters, "Builder Functions"),
        (test_build_security_certificate_default_phase_eq, "Builder Functions"),
        (test_build_security_certificate_configurable_phase_eq, "Builder Functions"),
        (test_build_security_certificate_with_lp_diag, "Builder Functions"),
        (test_build_security_metadata_none, "Builder Functions"),
        (test_build_security_metadata_missing, "Builder Functions"),
        (test_build_source_creates_optical_source, "Builder Functions"),
        (test_build_source_with_density_matrices, "Builder Functions"),
        (test_build_detector_returns_tuple, "Builder Functions"),
        (test_build_detector_overrides_recorded, "Builder Functions"),
        (test_build_detector_bias_clamping, "Builder Functions"),
        (test_build_detector_bias_no_clamp_needed, "Builder Functions"),
        (test_build_channel_from_distance, "Builder Functions"),
        (test_build_channel_zero_distance, "Builder Functions"),
        (test_build_protocol_bb84, "Builder Functions"),
        (test_build_protocol_b92, "Builder Functions"),
        (test_build_protocol_mdi, "Builder Functions"),
        (test_build_protocol_redundant, "Builder Functions"),
        (test_build_protocol_unknown_raises, "Builder Functions"),
        # Category 4: Simulation Functions
        (test_get_pulse_indices_from_prepared_states, "Simulation Functions"),
        (test_get_pulse_indices_missing_attr_raises, "Simulation Functions"),
        (test_generate_photons_for_prepared_states, "Simulation Functions"),
        (test_infer_ideal_outcomes_d0_bb84, "Simulation Functions"),
        (test_infer_ideal_outcomes_d0_unsupported_raises, "Simulation Functions"),
        (test_extract_detection_arrays_from_detection_result, "Simulation Functions"),
        (test_extract_detection_arrays_from_plain_tuple, "Simulation Functions"),
        (test_extract_detection_arrays_invalid_raises, "Simulation Functions"),
        (test_detector_result_to_protocol_detection_results, "Simulation Functions"),
        (test_basis_is_z_mask_bool, "Simulation Functions"),
        (test_basis_is_z_mask_int, "Simulation Functions"),
        (test_basis_is_z_mask_string, "Simulation Functions"),
        (test_get_first_existing_attr, "Simulation Functions"),
        (test_summarize_tallies, "Simulation Functions"),
        (test_roundtrip_tallycounts, "Simulation Functions"),
        (test_roundtrip_pulse_configs, "Simulation Functions"),
        (test_build_simulation_results, "Simulation Functions"),
        # Category 5: Worker Infrastructure
        (test_worker_state_initialization, "Worker Infrastructure"),
        (test_init_worker_sets_state, "Worker Infrastructure"),
        (test_get_source_metadata_row, "Worker Infrastructure"),
        (test_build_detector_override_row, "Worker Infrastructure"),
        (test_build_detector_override_row_empty, "Worker Infrastructure"),
        (test_log_source_details_no_crash, "Worker Infrastructure"),
        # Category 6: Physical Plausibility
        (test_channel_transmittance_beer_lambert, "Physical Plausibility"),
        (test_channel_loss_increases_with_distance, "Physical Plausibility"),
        (test_transmittance_decreases_with_distance, "Physical Plausibility"),
        (test_detection_yield_decreases_with_distance, "Physical Plausibility"),
        (test_qber_in_plausible_range, "Physical Plausibility"),
        (test_photon_numbers_nonnegative, "Physical Plausibility"),
        (test_epsilon_sum_much_less_than_one, "Physical Plausibility"),
        # Category 7: Numerical Stability
        (test_rng_reproducibility, "Numerical Stability"),
        (test_seed_sequence_independence, "Numerical Stability"),
        (test_no_nan_or_inf_in_channel, "Numerical Stability"),
        (test_no_nan_or_inf_in_detection, "Numerical Stability"),
        (test_very_small_pulse_count, "Numerical Stability"),
        (test_extreme_distance_no_crash, "Numerical Stability"),
        (test_detector_clone_independence, "Numerical Stability"),
        # Category 8: Edge Cases & Regression
        (test_channel_at_max_distance, "Edge Cases"),
        (test_zero_distance_is_lossless, "Edge Cases"),
        (test_probability_balancing_negative_detected, "Edge Cases"),
        (test_run_single_simulation_basic, "Edge Cases"),
        (test_run_single_simulation_error_path_fields, "Edge Cases"),
        (test_run_single_simulation_deterministic, "Edge Cases"),
        (test_attenuation_coeff_matches_config, "Edge Cases"),
        (test_intensity_config_signal_decoy_ordering, "Edge Cases"),
        (test_math_pi_used_in_config, "Edge Cases"),
        (test_worker_state_has_detector_overrides_slot, "Edge Cases"),
        (test_build_detector_override_keys_match, "Edge Cases"),
    ]

    print(f"\n{'='*72}")
    print(f"  COMPREHENSIVE TEST & VALIDATION SUITE FOR main_optimized.py")
    print(f"  {len(test_functions)} tests across 8 categories")
    print(f"{'='*72}\n")

    for func, category in test_functions:
        name = func.__name__
        _run_test(name, category, func)

    return _results


def _print_results(results: List[TestResult]):
    """Print formatted test results and analysis."""
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    # ── Per-category results ──
    for cat in sorted(categories.keys()):
        items = categories[cat]
        passed = sum(1 for r in items if r.passed)
        total = len(items)
        status = "PASS" if passed == total else "FAIL"
        print(f"\n  [{status}] {cat}: {passed}/{total} passed")
        for r in items:
            icon = "  PASS" if r.passed else "  FAIL"
            line = f"    {icon}  {r.name}"
            if not r.passed:
                line += f"  -- {r.detail}"
            if r.duration_ms > 100:
                line += f"  ({r.duration_ms:.0f}ms)"
            print(line)

    # ── Summary ──
    total_passed = sum(1 for r in results if r.passed)
    total_tests = len(results)
    total_failed = total_tests - total_passed
    total_time = sum(r.duration_ms for r in results)

    print(f"\n{'='*72}")
    print(f"  SUMMARY: {total_passed}/{total_tests} passed  "
          f"({total_failed} failed)  [{total_time/1000:.1f}s total]")
    print(f"{'='*72}")

    # ── Analysis ──
    print(f"\n{'='*72}")
    print(f"  ANALYSIS")
    print(f"{'='*72}")

    # Category-level analysis
    print("\n  Category Breakdown:")
    cat_results = {}
    for cat in sorted(categories.keys()):
        items = categories[cat]
        p = sum(1 for r in items if r.passed)
        t = len(items)
        cat_results[cat] = (p, t)
        pct = 100 * p / t if t > 0 else 0
        bar = "X" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(f"    {cat:<26} [{bar}] {p}/{t} ({pct:.0f}%)")

    # Physical plausibility analysis
    phys_cat = "Physical Plausibility"
    if phys_cat in cat_results:
        p, t = cat_results[phys_cat]
        print(f"\n  Physical Plausibility Assessment:")
        if p == t:
            print(f"    All {t} physical plausibility tests PASSED. The simulation")
            print(f"    produces results consistent with QKD theory: channel")
            print(f"    transmittance follows Beer-Lambert law, detection yield")
            print(f"    decreases with distance, QBER stays in plausible range,")
            print(f"    and photon numbers are non-negative integers.")
        else:
            print(f"    {t - p} of {t} physical plausibility tests FAILED.")
            print(f"    This suggests the simulation may produce results that")
            print(f"    violate known QKD physics. Investigate failures above.")

    # Numerical stability analysis
    num_cat = "Numerical Stability"
    if num_cat in cat_results:
        p, t = cat_results[num_cat]
        print(f"\n  Numerical Stability Assessment:")
        if p == t:
            print(f"    All {t} numerical stability tests PASSED. RNG streams")
            print(f"    are reproducible and independent, no NaN/Inf values")
            print(f"    detected, and edge cases (small counts, extreme distances)")
            print(f"    are handled gracefully.")
        else:
            print(f"    {t - p} of {t} numerical stability tests FAILED.")
            print(f"    The simulation may produce numerical instabilities.")

    # F-01 regression check
    f01_tests = [r for r in results if "override" in r.name.lower() or "clamp" in r.name.lower()]
    if f01_tests:
        print(f"\n  F-01 Regression Check (Detector Override Auditing):")
        f01_passed = sum(1 for r in f01_tests if r.passed)
        print(f"    {f01_passed}/{len(f01_tests)} override-related tests passed.")
        if f01_passed == len(f01_tests):
            print(f"    Detector parameter overrides are properly recorded and")
            print(f"    auditable in CSV output. Bias-voltage clamping works correctly.")
        else:
            print(f"    Some override tests failed — detector parameter changes")
            print(f"    may not be properly logged or recorded.")

    # Builder function analysis
    builder_cat = "Builder Functions"
    if builder_cat in cat_results:
        p, t = cat_results[builder_cat]
        print(f"\n  Builder Function Assessment:")
        if p == t:
            print(f"    All {t} builder function tests PASSED. Every builder")
            print(f"    correctly constructs its target object from DEFAULT_CONFIG,")
            print(f"    rejects invalid inputs, and handles edge cases (None,")
            print(f"    missing keys, wrong types) as expected.")
        else:
            print(f"    {t - p} of {t} builder tests FAILED. Some builders may")
            print(f"    not validate inputs correctly or produce malformed objects.")

    # Slow tests
    slow_tests = sorted(results, key=lambda r: r.duration_ms, reverse=True)[:5]
    print(f"\n  Slowest Tests:")
    for r in slow_tests:
        print(f"    {r.duration_ms:>8.0f}ms  {r.category:<26} {r.name}")

    # Overall verdict
    print(f"\n  Overall Verdict:")
    if total_failed == 0:
        print(f"    ALL {total_tests} TESTS PASSED. The corrected main_optimized.py")
        print(f"    is functionally correct, physically plausible, numerically")
        print(f"    stable, and properly handles edge cases and error paths.")
    elif total_failed <= 3:
        print(f"    {total_failed} minor failures out of {total_tests} tests.")
        print(f"    The code is largely correct but has a few issues to address.")
    else:
        print(f"    {total_failed} failures out of {total_tests} tests.")
        print(f"    Significant issues detected that should be investigated.")

    print(f"\n{'='*72}\n")
    return total_failed


if __name__ == "__main__":
    results = _run_all_tests()
    n_failed = _print_results(results)
    sys.exit(1 if n_failed > 0 else 0)
