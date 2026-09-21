# main_optimized.py
# -*- coding: utf-8 -*-
#
# TODO (F-24): Split this monolithic module into sub-modules:
#   config.py   – DEFAULT_CONFIG, DEFAULT_SWEEPS, builder functions
#   simulation.py – run_single_simulation, WorkerState
#   cli.py      – argparse entry point
# TODO (F-25): Migrate from dual dict/typed-object representation to
#   fully typed config objects, eliminating the dict→dataclass adapter layer.

import time
import json
import csv
import math                          # F-22: prefer math.pi over np.pi for constants
import argparse
import multiprocessing as mp
from functools import partial

# Testability hook: the dispatch loop (sequential / mp.Pool.imap_unordered)
# is extracted into main_optimized_dispatch so its synchronization and
# ordering contract can be exercised with deterministic stub workers,
# without needing the heavy qkd.* stack at import time.
from main_optimized_dispatch import dispatch_work_items
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Union, get_args, get_origin
import itertools
import copy
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)  # silence INFO logs for production sweeps

import numpy as np

from qkd.security_metadata import SourceSecurityMetadata

from qkd.datatypes import (
    ProtocolType, DetectorType, DoubleClickPolicy, DecoderArchitecture,
    SourceErrorModel, SecurityProof, ConfidenceBoundMethod, SimulationStatus,
    SourceStatisticsType, IntensityNode, IntensityConfig, AttenuationConfig,
    OpticalComponent, OpticalSourceConfig, DetectionConfig, ErrorCorrectionConfig,
    ProtocolParameters, PulseTypeConfig, PulseEnsembleConfig, TallyCounts,
    EpsilonAllocation, SecurityCertificate, SimulationResults,
)

from qkd.proofs.base import ProofMode
from qkd.proofs.lim2014 import Lim2014Proof
from qkd.proofs.optimization import optimize_pulses_for_distance
from qkd.sources import OpticalSource, PoissonSource, DensityMatrixSource
from qkd.sources import (
    photon_number_distribution,
    poisson_pn_array,
    thermal_pn_array,
    PN_TRUNCATION_DIM,
    SourceSimParams,
    build_source_sim_params,
)  # Analytical P(n) helpers — used by proof modules and future key-rate
    # pre-computation.  Keep these imports so that downstream refactors
    # (e.g., finite-key gain-curve pre-computation) can call them directly
    # from main_optimized without re-importing.
from qkd.params import load_lim2014_dedicated_params, load_lim2014_dwdm_params
from qkd.exceptions import (
    QKDException,
    ParameterValidationError,
    ConfigurationError,
    QKDSimulationError,
    LPFailureError,
    SimulationInterruptedError,
)
from qkd.channel import FiberChannel, MAX_DISTANCE_KM, MAX_FIBER_LOSS_DB_KM
from qkd.channel import (
    build_attenuation_config_from_sim_config,
    transmittance_scalar,
    transmittance_array,
    link_efficiency,
    ChannelSimParams,
    build_channel_sim_params,
)
from qkd.detectors import (
    SinglePhotonDetector,
    AfterpulseModel,
    DeadTimeModel,
    EntanglingDecoder,
    DetectionDiagnostics,
    DetectionResult,
    STATE_VERSION,
    DetectorSimParams,
    build_detector_sim_params,
    # Rogers et al. (2007) integration ----------------------------------
    # ``ROGERS_2007_POLICY`` is a module-level sentinel string (NOT a
    # DoubleClickPolicy enum value, because that enum lives in qkd.datatypes
    # and we deliberately do not mutate it).  Pass it as ``double_click_policy``
    # to enable the Rogers-style sequence-collapsing sifting rule.
    ROGERS_2007_POLICY,
    # Analytic helpers from Rogers et al. §3-4 (Eqs. 8, 9, 10-13, 15, 16, 17).
    # Used to cross-check the Monte-Carlo sifted-bit rate against the
    # paper's closed-form curves (Figs. 3-5).
    rogers_P_00,
    rogers_T_N,
    rogers_S,
    rogers_sifted_bit_rate,
    rogers_sbr_max,
    rogers_rho_tx_max,
)
from qkd.modulators import MZMConfig
from qkd.noise_models import ElectricalNoiseConfig
from qkd.io import safe_json_dumps, parse_json_strict, open_atomic_text

from qkd.protocols import (
    Protocol, BB84DecoyProtocol, B92Protocol, MDIQKDProtocol,
    RedundantTransmissionProtocol, DetectionResults, SiftingResults,
    BB84PreparedStates, B92PreparedStates, MDIPreparedStates,
    make_worker_rngs, sample_for_parameter_estimation,
)

# ============== CONFIGURATION ==============
DEFAULT_CONFIG = {
    "protocol_params": {
        "protocol": "BB84_DECOY",
        "detector_type": "SPD",
        "error_correction_efficiency": 1.16,
    },
    "protocol_runtime": {
        "protocol_class": "BB84DecoyProtocol",
        "alice_z_basis_prob": 0.5,
        "bob_z_basis_prob": 0.5,
        "z_basis_prob": 0.5,
        "double_click_policy": "DISCARD",
        "redundancy_M": 3,
        "use_entangling_decoder": False,
        "use_entangling_encoder": False,
        "codeword_mapping": None,
        "damping_parameter": None,
        "rotation_angle": None,
        "parameter_estimation_fraction": 0.1,
        "num_worker_rngs": 2,
    },
    "channel": {
        "fiber_loss_db_km": 0.2,
        "dispersion_parameter_ps_nm_km": 0.0,
    },
    "detector": {
        # In main_optimized.py, DEFAULT_CONFIG["detector"]:
        "det_eff_d0": 0.15, "det_eff_d1": 0.15,
        "dark_rate": 600.0,                    # was 600.0
        "dark_rate_d1": 1200.0,                 # was 1200.0
        "qber_intrinsic": 0.005,               # was 0.005
        "misalignment": 0.0,                # already 0
        "dead_time_ns": 10.0,                # was 10.0
        "jitter_fwhm_ns": 0.0,              # was 0.050
        "afterpulse_prob": 0.001,             # was 0.001
        "double_click_policy": "RANDOM", "detector_type": "SPD",
        "dead_time_model": "NON_PARALYZABLE", "afterpulse_model": "EXPONENTIAL",
        "afterpulse_lifetime_ns": 10.0,
        "temperature_k": 293.0, "bias_voltage": 50.0, "breakdown_voltage": 45.0,
        "ref_temperature_k": 293.0, "ref_bias_voltage": 50.0, "strict_mode": False,
    },
    "source": {
        "source_class": "optical", "pulse_period_ns": 10.0,
        "intensity_jitter": 0.0, "modulation_index": 0.0,  # WDM intensity modulation (needed for Rogers)
        "statistics_type": "POISSON", "error_model": "RANDOM_GAUSSIAN",
        "expected_decoder": "LOCAL", "assumed_double_click": "RANDOM",
        "intended_proof": "LIM_2014", "confidence_method": "GAUSSIAN",
        "source_fidelity": 0.995, "adversarial_block_size": 1000,
        "extinction_ratio": 80.0, "temperature_k": 298.15,
        "bandwidth_hz": 1e9, "driver_load_resistance_ohm": 50.0,
        "dc_photocurrent_a": 1e-6, "v_pi": 3.5,
        # F-22: Use math.pi instead of np.pi for a pure-Python constant
        "phi_bias": math.pi / 2.0, "max_intensity_mu": 1.0,
        "preferred_lp_solver": "highs", "N_channels": 1,
        "is_bidirectional": False, "use_small_angle_approximation": True,
        "use_linear_modulation_approximation": False,
        "ideal_emission_probability": 1.0,
        "mzm": {
            "extinction_ratio_db": 80.0,
            "v_pi": 3.5,
            "bias_voltage": 1.75,
            "rf_amplitude_v": 3.5,
        },
        "electrical_noise": {
            "voltage_std_v": 0.05,
            "bandwidth_hz": 1e9,
        },
        "security_metadata": None,
        "pulses": {
            "signal": {"mu": 0.5, "prob": 0.6},
            "decoy":  {"mu": 0.1, "prob": 0.2},
            "vacuum": {"mu": 0.0, "prob": 0.2},
        },
        "density_matrices": {},
    },
    "protocol": {
        # F-17: Epsilon values are validated in build_epsilon_allocation()
        # to ensure they are positive and sum below eps_total.
        "epsilons": {
            # Composable-security convention: each sub-epsilon <= eps_sec.
            # Matches the Lim2014 factory (params.py line 1591) which sets all
            # four to 1e-7. Using 1e-10 here keeps eps_sec=1e-9 as the budget
            # headroom and removes the need for the MA2005/WANG2005 runtime
            # auto-rescale (lines 2477-2541).
            "eps_sec": 1e-9, "eps_cor": 1e-10, "eps_pe": 1e-10,
            "eps_smooth": 1e-10, "eps_pa": 1e-10, "eps_phase_est": 1e-10,
        }
    },
    "simulation": {
        "apply_statistical_noise": True, "sampling_cap_per_pulse_type": 200000,
        "rng_seed": 12345, "min_pulses_log": 4, "max_pulses_log": 12,
        "distance_start_km": 0.0, "distance_stop_km": 150.0, "distance_points": 16,
    },
}

# ============== SWEEP CONFIGURATION ==============
# F-15: The Cartesian product of all sweep lists can produce a very large number
# of combinations. A warning is emitted in run_and_save_csv() when the count
# exceeds a configurable threshold (default 10 000).
DEFAULT_SWEEPS = {
    #"source.pulses.signal.mu": [0.3, 0.5, 0.8],
    #"source.pulses.decoy.mu": [0.05, 0.1, 0.2],
    #"detector.det_eff_d0": [0.10, 0.15, 0.20],
    #"detector.det_eff_d1": [0.10, 0.15, 0.20],
    #"detector.dark_rate": [1e-7, 1e-6, 1e-5],
    #"detector.bias_voltage": [44.0, 50.0],
    #"detector.temperature_k": [77.0, 293.0],
    #"detector.afterpulse_lifetime_ns": [10.0, 100.0, 500.0],
    #"detector.qber_intrinsic": [0.005, 0.01, 0.02],
    #"detector.misalignment": [0.002, 0.005, 0.01],
    #"source.statistics_type": ["POISSON", "THERMAL"],
    #"source.intended_proof": ["LIM_2014", "MA_2005", "WANG_2005", "TIGHT"],
    #"source.confidence_method": ["GAUSSIAN", "HOEFFDING", "CLOPPER_PEARSON"],
    #"source.intended_proof": ["LIM_2014", "MA_2005", "WANG_2005", "TIGHT"],
    #"source.ideal_emission_probability": [0.95, 0.99, 1.0],
    #"source.use_linear_modulation_approximation": [False, True],
    #"source.N_channels": [2, 4],
    #"source.is_bidirectional": [False, True],
    #"source.density_matrices": [None, {"signal": [[1.0, 0.0], [0.0, 0.0]], "decoy": [[1.0, 0.0], [0.0, 0.0]], "vacuum": [[1.0, 0.0], [0.0, 0.0]]}],
    #"detector.detector_type": ["SPD", "PNRD", "SNSPD"],
    # F-28: ``DeadTimeModel.PARALYZABLE`` is deprecated per Rogers et al.
    # (2007): individual SPADs are non-paralyzable.  Basis-level
    # paralyzability is recovered by selecting ``ROGERS_2007_POLICY`` for
    # the double-click policy, not by setting a per-detector paralyzable
    # flag.  We therefore drop ``PARALYZABLE`` from the sweep (it would
    # just duplicate the ``NON_PARALYZABLE`` run with a deprecation
    # warning) and add ``rogers_2007`` to the double-click policy sweep.
    #"detector.dead_time_model": ["NON_PARALYZABLE"],
    #"detector.afterpulse_model": ["EXPONENTIAL", "GEOMETRIC"],
    #"detector.double_click_policy": ["RANDOM", "DISCARD", "rogers_2007"],
    #"channel.dispersion_parameter_ps_nm_km": [0.0, 17.0],
    #"detector.strict_mode": [False, True],
    #"protocol_runtime.protocol_class": ["BB84DecoyProtocol", "RedundantTransmissionProtocol"],
    #"protocol_runtime.use_entangling_decoder": [False, True],
    #"protocol_runtime.redundancy_M": [2, 3],
}


# ============== DETECTOR PRESETS ==============
# Physically realistic parameters for each detector type.
# Applied automatically when detector_type is changed via sweep or CLI.
# Sources:
#   SPD  (InGaAs SPAD):  Hadke et al., NJP 2016; Korzh et al., PRL 2015
#   SNSPD:               You et al., Nature 2023; Marsili et al., Nat. Photon. 2013
#   PNRD:                Chen et al., PRL 2022; Humphreys et al., Nature 2020
DETECTOR_PRESETS = {
    "SPD": {
        "det_eff_d0": 0.15,               # InGaAs SPAD: 10-25% typical
        "det_eff_d1": 0.15,
        "dark_rate": 600.0,                # ~600 Hz after gating
        "dark_rate_d1": 1200.0,           # D1 slightly higher (afterpulsing)
        "afterpulse_prob": 0.008,         # ~0.8% afterpulsing
        "afterpulse_lifetime_ns": 20.0,   # 20 ns decay
        "dead_time_ns": 20.0,             # 20 ns gate-to-gate
        "jitter_fwhm_ns": 0.300,          # 300 ps (SPAD timing)
    },
    "SNSPD": {
        "det_eff_d0": 0.75,               # SNSPD: 60-90% typical, 75% conservative
        "det_eff_d1": 0.75,
        "dark_rate": 1.0,                # ~1 Hz (extremely low)
        "dark_rate_d1": 1.0,
        "afterpulse_prob": 0.0,           # No afterpulsing
        "afterpulse_lifetime_ns": 0.0,
        "dead_time_ns": 10.0,             # 10 ns recovery
        "jitter_fwhm_ns": 0.020,          # 20 ps (SNSPD timing)
    },
    "PNRD": {
        "det_eff_d0": 0.45,               # PNRD/SNSPD array: 30-60%
        "det_eff_d1": 0.45,
        "dark_rate": 1000.0,                # ~1 kHz (array adds counts)
        "dark_rate_d1": 1000.0,
        "afterpulse_prob": 0.0,           # No afterpulsing (transition-edge)
        "afterpulse_lifetime_ns": 0.0,
        "dead_time_ns": 50.0,             # 50 ns (slower recovery)
        "jitter_fwhm_ns": 0.100,          # 100 ps
    },
}


def set_nested_value(d: Dict, key_path: str, value: Any, *, strict: bool = False):
    """Sets a value in a nested dictionary using a dot-separated string.

    Args:
        strict: When True, raise ConfigurationError if any intermediate key
            does not already exist. This prevents silently creating typo'd
            key paths (F-10).
    """
    if not key_path or not isinstance(key_path, str):
        raise ConfigurationError(
            "Sweep override key_path must be a non-empty dotted string.",
            context={"key_path": key_path},
        )

    keys = key_path.split('.')
    current = d
    for key in keys[:-1]:
        if not isinstance(current, dict):
            raise ConfigurationError(
                "Cannot apply sweep override through a non-dictionary config node.",
                context={"key_path": key_path, "blocked_at": key},
            )
        if strict and key not in current:
            raise ConfigurationError(
                "Sweep override key_path references a missing intermediate key.",
                context={"key_path": key_path, "missing_key": key},
            )
        current = current.setdefault(key, {})
        if not isinstance(current, dict):
            raise ConfigurationError(
                "Cannot apply sweep override because an intermediate config value is not a dictionary.",
                context={"key_path": key_path, "blocked_at": key, "value_type": type(current).__name__},
            )
    current[keys[-1]] = value


# ============== HELPER FUNCTIONS ==============
def _resolve_enum_type(annotation):
    try:
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return annotation
    except TypeError:
        pass
    origin = get_origin(annotation)
    if origin is Union:
        for arg in get_args(annotation):
            try:
                if isinstance(arg, type) and issubclass(arg, Enum):
                    return arg
            except TypeError:
                continue
    return None

def _build_dataclass_config(cls, data):
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Expected a dictionary to build {cls.__name__}, got {type(data).__name__}.",
            context={"dataclass": cls.__name__, "value_type": type(data).__name__},
        )
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        enum_cls = _resolve_enum_type(f.type)
        if enum_cls is not None and isinstance(value, str):
            value = parse_enum(enum_cls, value, param_name=f.name)
        kwargs[f.name] = value
    try:
        return cls(**kwargs)
    except ParameterValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"Failed to build {cls.__name__} from configuration.",
            context={"dataclass": cls.__name__, "keys": sorted(data.keys())},
            cause=exc,
        ) from exc

def build_pulse_type_config(raw_pulse):
    return _build_dataclass_config(PulseTypeConfig, raw_pulse)

def parse_enum(enum_cls, value: str, *, param_name: Optional[str] = None):
    if isinstance(value, enum_cls):
        return value

    if isinstance(value, str):
        candidate = value.strip()
        candidate_name = candidate.upper()
        candidate_value = candidate.lower()
        for member in enum_cls:
            if member.name == candidate_name or str(member.value).lower() == candidate_value:
                return member
    else:
        try:
            return enum_cls(value)
        except (TypeError, ValueError):
            pass

    valid = [f"{member.name}/{member.value}" for member in enum_cls]
    raise ParameterValidationError(
        f"Invalid value {value!r} for {enum_cls.__name__}. Use one of: {valid}.",
        param_name=param_name or enum_cls.__name__,
        param_value=value,
        context={"enum": enum_cls.__name__, "valid_values": valid},
    )

def normalize_detector_config(det_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize detector config dict, coercing string values to enums.

    Special case: ``double_click_policy == ROGERS_2007_POLICY`` is a
    module-level sentinel string defined in ``qkd.detectors``, not a
    ``DoubleClickPolicy`` enum member (that enum lives in ``qkd.datatypes``
    and we deliberately do not mutate it).  We pass the sentinel through
    unchanged so the detector can dispatch on it via ``_is_rogers_policy``.
    """
    normalized = dict(det_cfg)
    enum_fields = {
        "double_click_policy": DoubleClickPolicy,
        "detector_type": DetectorType,
        "dead_time_model": DeadTimeModel,
        "afterpulse_model": AfterpulseModel,
    }
    for key, enum_cls in enum_fields.items():
        if key in normalized and normalized[key] is not None:
            value = normalized[key]
            # Rogers et al. (2007) sifting policy sentinel: pass through.
            if key == "double_click_policy" and value == ROGERS_2007_POLICY:
                continue
            normalized[key] = parse_enum(enum_cls, value)
    return normalized


def _log_qkd_exception(exc: QKDException, *, level: int = logging.ERROR) -> None:
    exc.log(logger, level=level)

def _as_qkd_exception(
    exc: BaseException,
    message: str,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> QKDException:
    """Wrap a bare exception as the most appropriate QKDException subclass.

    Standardised error-mapping helper used by all catch-blocks (F-23).
    """
    if isinstance(exc, QKDException):
        return exc
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        return ConfigurationError(message, context=context, cause=exc)
    return QKDSimulationError(message, context=context, cause=exc)


def build_epsilon_allocation(config: Dict[str, Any]) -> EpsilonAllocation:
    """Build and validate epsilon allocation from config.

    F-17: Validates that all epsilon values are positive and their sum is
    less than 1 (the total security budget).
    """
    eps = config["protocol"]["epsilons"]
    allocation = EpsilonAllocation(
        eps_sec=eps["eps_sec"], eps_cor=eps["eps_cor"], eps_pe=eps["eps_pe"],
        eps_smooth=eps["eps_smooth"], eps_pa=eps["eps_pa"], eps_phase_est=eps["eps_phase_est"],
    )
    # F-17: Validate epsilon allocation
    total_eps = (
        allocation.eps_sec + allocation.eps_cor + allocation.eps_pe +
        allocation.eps_smooth + allocation.eps_pa + allocation.eps_phase_est
    )
    if total_eps >= 1.0:
        raise ParameterValidationError(
            f"Total epsilon allocation ({total_eps:.2e}) must be less than 1.",
            param_name="protocol.epsilons",
            param_value=total_eps,
            context={"total_epsilon": total_eps},
        )
    for field_name in ("eps_sec", "eps_cor", "eps_pe", "eps_smooth", "eps_pa", "eps_phase_est"):
        val = getattr(allocation, field_name)
        if val <= 0:
            raise ParameterValidationError(
                f"Epsilon value {field_name}={val} must be positive.",
                param_name=f"protocol.epsilons.{field_name}",
                param_value=val,
            )
    return allocation

def build_pulse_configs(config: Dict[str, Any]) -> Tuple[PulseTypeConfig, ...]:
    """Build pulse type configs directly from config (F-18: dead DEBUG_PULSE_OVERRIDE removed)."""
    pulse_configs: List[PulseTypeConfig] = []
    for name, p_data in config["source"]["pulses"].items():
        mu, prob = p_data["mu"], p_data["prob"]
        pulse_configs.append(PulseTypeConfig(name=name, mean_photon_number=mu, probability=prob))
    return tuple(pulse_configs)

# F-18: apply_debug_pulse_override() removed.  The function was dead code:
# DEBUG_PULSE_OVERRIDE was always False, and its single call site
# (run_single_simulation line ~1124) is now also removed.  If a forced-pulse
# debug mode is needed in the future, implement it behind an explicit CLI flag.


def build_intensity_config(config: Dict[str, Any]) -> IntensityConfig:
    """Build IntensityConfig from config dict (pre-construction adapter).

    This function is a config-dict adapter used ONLY when the OpticalSource
    object has not yet been constructed (e.g., during ProtocolParameters
    bootstrap in run_and_save_csv). Once the source object is available,
    prefer ``source.intensity_config`` which:
      - Uses role-based pulse-name lookup (finds "signal" by name, not
        by hardcoded index).
      - Supports variable decoy counts (not limited to 3 pulse types).
      - Reads from the validated, frozen source state rather than the
        mutable config dict.

    NOTE: The hardcoded pulse names ("signal", "decoy", "vacuum") assume
    exactly 3 pulse types. This will break for configs with more decoys
    or different naming conventions. Use ``source.intensity_config`` for
    dynamic role-based mapping.
    """
    pulses = config["source"]["pulses"]
    signal = IntensityNode(mu=float(pulses["signal"]["mu"]), probability=float(pulses["signal"]["prob"]))
    decoys = [
        IntensityNode(mu=float(pulses["decoy"]["mu"]), probability=float(pulses["decoy"]["prob"])),
        IntensityNode(mu=float(pulses["vacuum"]["mu"]), probability=float(pulses["vacuum"]["prob"])),
    ]
    return IntensityConfig(signal=signal, decoys=decoys)


def build_intensity_config_from_source(source: OpticalSource) -> IntensityConfig:
    """Build IntensityConfig from an already-constructed OpticalSource.

    This is the preferred path when the source object is available. It
    delegates to ``source.intensity_config`` which uses role-based pulse
    name lookup ("signal" by name, fallback to index 0) and supports
    variable decoy counts.

    Parameters
    ----------
    source : OpticalSource
        A fully constructed and validated source object.

    Returns
    -------
    IntensityConfig
        The source's intensity configuration, derived from its validated
        pulse_configs.
    """
    return source.intensity_config

def build_attenuation_config(distance_km: float, config: Dict[str, Any]) -> AttenuationConfig:
    """Build AttenuationConfig from config dict — delegates to channel.py.

    This function is now a thin wrapper around
    ``build_attenuation_config_from_sim_config`` from ``qkd.channel.py``,
    which is the single source-of-truth adapter for converting the
    main_optimized config dict format into an AttenuationConfig.

    Delegating to channel.py ensures that all channel parameter
    validation (non-negative alpha, MAX_FIBER_LOSS_DB_KM bounds) is
    performed in one module, eliminating the duplicated validation
    that previously existed in main_optimized.py.
    """
    return build_attenuation_config_from_sim_config(distance_km, config)

def build_optical_source_config(config: Dict[str, Any]) -> OpticalSourceConfig:
    source_cfg = config["source"]
    # ── Derive source_rate from config ──
    # OpticalSourceConfig does NOT have pulse_period_ns as a dataclass
    # field — it is a @property computed from source_rate
    # (1e9 / source_rate).  We only pass source_rate; pulse_period_ns
    # is derived automatically.  If the config dict specifies
    # pulse_period_ns instead of source_rate, we compute source_rate
    # from it so that the property returns the correct value.
    if "source_rate" in source_cfg and source_cfg["source_rate"] is not None:
        source_rate = float(source_cfg["source_rate"])
    elif "pulse_period_ns" in source_cfg and source_cfg["pulse_period_ns"] is not None:
        pulse_period_ns = float(source_cfg["pulse_period_ns"])
        source_rate = 1e9 / pulse_period_ns if pulse_period_ns > 0 else 1e8
    else:
        raise ConfigurationError(
            "source config must contain either 'source_rate' or 'pulse_period_ns'.",
            context={"section": "source", "available_keys": sorted(source_cfg.keys())},
        )
    pulse_configs = build_pulse_configs(config)
    return OpticalSourceConfig(
        source_rate=source_rate,
        pulse_configs=pulse_configs,
        statistics_type=parse_enum(SourceStatisticsType, source_cfg.get("statistics_type", SourceStatisticsType.POISSON)),
        error_model=parse_enum(SourceErrorModel, source_cfg.get("error_model", SourceErrorModel.RANDOM_GAUSSIAN)),
        intensity_jitter=source_cfg.get("intensity_jitter", 0.0),
        modulation_index=source_cfg.get("modulation_index", 0.0),
        N_channels=source_cfg.get("N_channels", 1),
        use_small_angle_approximation=source_cfg.get("use_small_angle_approximation", True),
        use_linear_modulation_approximation=source_cfg.get("use_linear_modulation_approximation", False),
        ideal_emission_probability=source_cfg.get("ideal_emission_probability", 1.0),
        adversarial_block_size=source_cfg.get("adversarial_block_size", 1000),
        is_bidirectional=source_cfg.get("is_bidirectional", False),
        mzm=build_mzm_config(source_cfg),
        electrical_noise=build_electrical_noise_config(source_cfg),
    )

# F-21: build_optical_component() removed.  It existed solely for coverage
# probing of the OpticalComponent dataclass.  Coverage of dataclass
# construction should be verified in the project's test suite, not in
# production code.

def build_mzm_config(source_cfg: Dict[str, Any]) -> Optional[MZMConfig]:
    raw_mzm = source_cfg.get("mzm")
    if isinstance(raw_mzm, MZMConfig):
        return raw_mzm
    if raw_mzm is None:
        return None

    mzm_cfg = dict(raw_mzm)
    extinction_ratio_db = mzm_cfg.get(
        "extinction_ratio_db",
        source_cfg.get("extinction_ratio"),
    )
    if extinction_ratio_db is not None:
        extinction_ratio_db = float(extinction_ratio_db)

    return MZMConfig(
        v_pi=float(mzm_cfg.get("v_pi", source_cfg.get("v_pi", 3.5))),
        phi_bias=float(mzm_cfg.get("phi_bias", source_cfg.get("phi_bias", math.pi / 2.0))),
        max_intensity_mu=float(
            mzm_cfg.get("max_intensity_mu", source_cfg.get("max_intensity_mu", 1.0))
        ),
        extinction_ratio_db=extinction_ratio_db,
    )

def build_electrical_noise_config(source_cfg: Dict[str, Any]) -> Optional[ElectricalNoiseConfig]:
    raw_noise = source_cfg.get("electrical_noise")
    if isinstance(raw_noise, ElectricalNoiseConfig):
        return raw_noise
    if raw_noise is None:
        return None

    noise_cfg = dict(raw_noise)
    monitor_photocurrent_a = noise_cfg.get(
        "monitor_photocurrent_a",
        noise_cfg.get("dc_photocurrent_a", source_cfg.get("dc_photocurrent_a")),
    )
    if monitor_photocurrent_a is not None:
        monitor_photocurrent_a = float(monitor_photocurrent_a)

    return ElectricalNoiseConfig(
        temperature_k=float(noise_cfg.get("temperature_k", source_cfg.get("temperature_k", 298.15))),
        bandwidth_hz=float(noise_cfg.get("bandwidth_hz", source_cfg.get("bandwidth_hz", 1e9))),
        load_resistance_ohm=float(
            noise_cfg.get(
                "load_resistance_ohm",
                noise_cfg.get(
                    "driver_load_resistance_ohm",
                    source_cfg.get("driver_load_resistance_ohm", 50.0),
                ),
            )
        ),
        monitor_photocurrent_a=monitor_photocurrent_a,
    )

def build_detection_config(config: Dict[str, Any]) -> DetectionConfig:
    """Build a DetectionConfig with averaged efficiency and dark-count rate.

    F-20: The averaging-then-override pattern is intentional: from_config()
    requires a single (efficiency, dark_count_rate) pair, while the full
    simulation needs per-detector (d0/d1) values.  The averaged values are
    only used to bootstrap the detector; per-detector values are applied
    immediately after construction in build_detector().
    """
    det_cfg = normalize_detector_config(config["detector"])
    avg_eff = 0.5 * (float(det_cfg["det_eff_d0"]) + float(det_cfg["det_eff_d1"]))
    avg_dark = float(det_cfg["dark_rate"])
    if det_cfg.get("dark_rate_d1", None) is not None:
        avg_dark = 0.5 * (float(det_cfg["dark_rate"]) + float(det_cfg["dark_rate_d1"]))
    detector_type = det_cfg.get(
        "detector_type",
        config.get("protocol_params", {}).get("detector_type", DetectorType.SPD),
    )
    return DetectionConfig(
        efficiency=avg_eff, dark_count_rate=avg_dark,
        detector_type=parse_enum(DetectorType, detector_type),
    )

def build_error_correction_config(config: Dict[str, Any]) -> ErrorCorrectionConfig:
    return ErrorCorrectionConfig(efficiency=float(config["protocol_params"]["error_correction_efficiency"]))

def build_protocol_parameters(
    distance_km: float,
    config: Dict[str, Any],
    source: Optional[OpticalSource] = None,
) -> ProtocolParameters:
    """Build ProtocolParameters from config and optionally an OpticalSource.

    When ``source`` is provided, the intensities are derived from the
    source object via ``build_intensity_config_from_source(source)``,
    which uses role-based pulse-name lookup and supports variable decoy
    counts. When ``source`` is None, the legacy config-dict adapter
    ``build_intensity_config(config)`` is used (hardcoded 3-pulse-type
    assumption).

    The source also provides ``optical`` (OpticalSourceConfig), so when
    the source is available, ``build_optical_source_config()`` is bypassed
    and ``source.config`` is used directly.
    """
    if source is not None:
        intensities = build_intensity_config_from_source(source)
        optical = build_optical_source_config(config)   # always rebuild from config dict
    else:
        intensities = build_intensity_config(config)
        optical = build_optical_source_config(config)
    return ProtocolParameters(
        protocol=parse_enum(ProtocolType, config["protocol_params"]["protocol"]),
        intensities=intensities,
        attenuation=build_attenuation_config(distance_km, config),
        optical=optical,
        detection=build_detection_config(config),
        error_correction=build_error_correction_config(config),
    )

def build_security_certificate(
    config: Dict[str, Any],
    lp_solver_diagnostics: Optional[Dict[str, Any]] = None,
) -> SecurityCertificate:
    """Build a SecurityCertificate from config.

    F-16: assumed_phase_equals_bit_error is now read from config instead of
    being hardcoded to True.  Defaults to True for backward compatibility.
    """
    assumed_phase_eq = config.get("protocol", {}).get(
        "assumed_phase_equals_bit_error", True
    )
    return SecurityCertificate(
        proof_name=parse_enum(SecurityProof, config["source"]["intended_proof"]),
        confidence_bound_method=parse_enum(ConfidenceBoundMethod, config["source"]["confidence_method"]),
        assumed_phase_equals_bit_error=bool(assumed_phase_eq),
        epsilon_allocation=build_epsilon_allocation(config),
        lp_solver_diagnostics=lp_solver_diagnostics,
    )

def _build_security_metadata(source_cfg: Dict[str, Any]) -> Optional[SourceSecurityMetadata]:
    """Attempt to build SourceSecurityMetadata from source config; return None on failure."""
    raw = source_cfg.get("security_metadata")
    if raw is None:
        return None

    if not hasattr(SourceSecurityMetadata, "from_dict"):
        metadata_error = ConfigurationError(
            "source.security_metadata is configured, but SourceSecurityMetadata.from_dict is not available; continuing without metadata.",
            context={"section": "source.security_metadata"},
        )
        _log_qkd_exception(metadata_error, level=logging.WARNING)
        return None

    try:
        return SourceSecurityMetadata.from_dict(raw)
    except QKDException as exc:
        _log_qkd_exception(exc, level=logging.WARNING)
    except (TypeError, ValueError, KeyError) as exc:
        metadata_error = ConfigurationError(
            "Failed to parse source.security_metadata; continuing without metadata.",
            context={"section": "source.security_metadata"},
            cause=exc,
        )
        _log_qkd_exception(metadata_error, level=logging.WARNING)
    return None


def build_source(config: Dict[str, Any]) -> OpticalSource:
    """Build optical source via OpticalSource.create() for clean sources.py integration.

    Uses build_optical_source_config() as the single source-of-truth adapter from
    the main config dict format to OpticalSourceConfig, then delegates to
    OpticalSource.create() which handles DensityMatrixSource / PoissonSource
    dispatching, density-matrix validation (via _coerce_density_matrices), and
    security metadata wiring internally.
    """
    source_cfg = config["source"]

    # Build OpticalSourceConfig — single adapter shared with build_protocol_parameters()
    optical_config = build_optical_source_config(config)

    # Build optional security metadata (graceful degradation on failure)
    security_metadata = _build_security_metadata(source_cfg)

    # Pass raw density_matrices dict directly; sources.py's _coerce_density_matrices()
    # performs full Hermitian / trace / PSD validation that the old build_density_matrices()
    # did not.
    density_matrices = source_cfg.get("density_matrices") or None

    return OpticalSource.create(
        optical_config,
        density_matrices=density_matrices,
        security_metadata=security_metadata,
    )

def log_source_details(source, level: int = logging.DEBUG) -> None:
    """Log source details using query methods from sources.py.

    F-26: Removed update_internal_tallies() call that was mutating source
    state for coverage purposes.  Removed sample_pulse_indices() coverage
    probe that created a throwaway RNG and sampled from the source.
    """
    if not logger.isEnabledFor(level):
        return

    logger.log(level, f"Source type: {type(source).__name__}")
    logger.log(level, f"Pulse names: {source.pulse_names()}")
    logger.log(level, f"Base μ values: {source.base_mean_photon_numbers()}")
    logger.log(level, f"Pulse probabilities: {source.pulse_probabilities()}")
    logger.log(level, f"Ideal emission probability: {source.ideal_emission_probability}")
    logger.log(level, f"Statistics type: {source.statistics_type.value}")
    logger.log(level, f"Error model: {source.error_model.value}")
    logger.log(level, f"Modulation index: {source.modulation_index}")
    logger.log(level, f"Intensity jitter: {source.intensity_jitter}")
    logger.log(level, f"N_channels: {source.N_channels}")
    # Access config.is_bidirectional directly to avoid the deprecated
    # property warning on OpticalSource.is_bidirectional.
    logger.log(level, f"Is bidirectional: {source.config.is_bidirectional}")
    logger.log(level, f"Use small angle approximation: {source.use_small_angle_approximation}")
    logger.log(level, f"Use linear modulation approximation: {source.use_linear_modulation_approximation}")
    logger.log(level, f"Adversarial block size: {source.adversarial_block_size}")
    logger.log(level, f"Pulse period (ns): {source.pulse_period_ns}")
    logger.log(level, f"Source rate (Hz): {source.source_rate}")

    if source.mzm is not None:
        logger.log(level, f"MZM enabled: {source.mzm}")
    else:
        logger.log(level, f"MZM: disabled (None)")

    if source.electrical_noise is not None:
        noise_std = source.electrical_noise.total_voltage_std()
        logger.log(level, f"Electrical noise: {source.electrical_noise}, total_std={noise_std:.6f}V")
    else:
        logger.log(level, f"Electrical noise: disabled (None)")

    if source.security_metadata is not None:
        logger.log(level, f"Security metadata: enabled, total_sent={source.security_metadata.total_sent}")
    else:
        logger.log(level, f"Security metadata: disabled (None)")

    for name in source.pulse_names():
        mu = source.get_mean_photon_number_by_name(name)
        prob = source.get_pulse_probability_by_name(name)
        logger.log(level, f"  Pulse '{name}': μ={mu:.6f}, p={prob:.6f}")

    logger.log(level, f"Pulse ensemble: {source.pulse_ensemble}")
    logger.log(level, f"Intensity config: {source.intensity_config}")

    if isinstance(source, DensityMatrixSource) and source.density_matrices:
        logger.log(level, f"Using density matrices for: {list(source.density_matrices.keys())}")
        for name in source.pulse_names():
            num_probs = source.number_probabilities_for_pulse(name)
            dm = source.density_matrix_for_pulse(name)
            logger.log(level, f"  DM for '{name}': shape={dm.shape}, photon probs (first 5)={num_probs[:5]}")


def _build_detector_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build a flat config dict for SinglePhotonDetector.from_config_dict().

    This replaces the previous two-step pattern:
      1. build_detection_config() -> DetectionConfig (only 3 fields)
      2. from_config() + 10 setattr overrides for d0/d1/overbias params

    The new pattern delegates ALL parameter handling to detectors.py's
    ``from_config_dict()`` which validates enums, rejects unknown keys,
    and runs ``_validate_params()`` in one atomic step.  No post-construction
    setattr overrides are needed — every detector parameter is passed via
    the config dict and validated by ``_validate_params()`` during
    construction.

    The Geiger-mode guard (SPAD: bias > breakdown) is now handled entirely
    by ``SinglePhotonDetector._validate_params()`` (which raises on invalid
    combos for SPD type), eliminating the duplicated guard that previously
    existed in main_optimized.py.

    Mapping from the nested config dict format to the flat detector dict:
    - ``config["detector"]["det_eff_d0"]`` -> ``det_eff_d0``
    - ``config["detector"]["det_eff_d1"]`` -> ``det_eff_d1``
    - ``config["detector"]["dark_rate"]`` -> ``dark_rate``
    - ``config["detector"]["dark_rate_d1"]`` -> ``dark_rate_d1``
    - All other detector fields are passed through as-is after enum
      normalization.
    """
    raw_det = dict(config["detector"])

    # ── Normalize enum-valued fields ──
    # This replaces normalize_detector_config() for the from_config_dict()
    # path.  ``from_config_dict()`` itself also normalizes enums, but we
    # pre-normalize here so that the overrides dict (returned for CSV
    # auditability) contains enum members, not raw strings.
    enum_fields = {
        "double_click_policy": DoubleClickPolicy,
        "detector_type": DetectorType,
        "dead_time_model": DeadTimeModel,
        "afterpulse_model": AfterpulseModel,
    }
    normalized = dict(raw_det)
    for key, enum_cls in enum_fields.items():
        if key in normalized and normalized[key] is not None:
            value = normalized[key]
            # Rogers et al. (2007) sifting policy sentinel: pass through.
            if key == "double_click_policy" and value == ROGERS_2007_POLICY:
                continue
            if isinstance(value, enum_cls):
                continue
            normalized[key] = parse_enum(enum_cls, value)

    # ── Convert numeric fields to proper types ──
    numeric_fields = {
        "det_eff_d0": float,
        "det_eff_d1": float,
        "dark_rate": float,
        "qber_intrinsic": float,
        "misalignment": float,
        "dead_time_ns": float,
        "jitter_fwhm_ns": float,
        "afterpulse_prob": float,
        "afterpulse_lifetime_ns": float,
        "bias_voltage": float,
        "breakdown_voltage": float,
        "temperature_k": float,
        "ref_temperature_k": float,
        "ref_bias_voltage": float,
    }
    for key, conv in numeric_fields.items():
        if key in normalized and normalized[key] is not None:
            normalized[key] = conv(normalized[key])

    # ── Handle optional fields ──
    # dark_rate_d1: if absent, from_config_dict() uses the default (None)
    # which means "same as dark_rate" inside the detector.
    if "dark_rate_d1" in normalized and normalized["dark_rate_d1"] is not None:
        normalized["dark_rate_d1"] = float(normalized["dark_rate_d1"])

    # strict_mode: ensure bool
    if "strict_mode" in normalized:
        normalized["strict_mode"] = bool(normalized["strict_mode"])

    # ── Remove non-detector keys that from_config_dict() would reject ──
    # The config dict may contain keys like "dispersion_parameter_ps_nm_km"
    # or other channel/source params that are NOT valid SinglePhotonDetector
    # fields.  from_config_dict() rejects unknown keys, so we must strip
    # them here.
    allowed_keys = {f.name for f in fields(SinglePhotonDetector) if f.init}
    # Allow recognized constructor kwargs that are NOT dataclass fields.
    # These are extracted by build_detector() and passed via from_config_dict(**extra_kwargs).
    allowed_keys.update({
        "flip_prob_override",
        "allow_xor_flip_formula",
    })
    clean_dict = {k: v for k, v in normalized.items() if k in allowed_keys}

    # ── Log stripped keys for auditability ──
    stripped = sorted(set(normalized.keys()) - allowed_keys)
    if stripped:
        logger.debug(
            "Stripped non-detector keys from config dict (not valid "
            "SinglePhotonDetector fields): %s",
            stripped,
        )

    return clean_dict


def build_detector(config: Dict[str, Any]) -> Tuple[SinglePhotonDetector, Dict[str, Any]]:
    """Build the main simulation detector via from_config_dict().

    Previous approach (removed):
      1. ``build_detection_config()`` produced a ``DetectionConfig`` with only
         3 fields (efficiency, dark_count_rate, detector_type), losing all
         per-detector and overbias parameters.
      2. ``SinglePhotonDetector.from_config()`` created a detector from that
         minimal config, then 10 ``setattr`` overrides patched in the missing
         per-detector efficiencies, dark rates, and overbias parameters —
         bypassing ``_validate_params()`` and creating a fragile,
         hard-to-audit construction path.
      3. A duplicated Geiger-mode guard (bias > breakdown for SPD) was
         maintained in main_optimized.py, separate from the one in
         detectors.py's ``_validate_params()``.

    New approach:
      ``_build_detector_config_dict()`` assembles a flat dict with ALL
      detector parameters (per-detector efficiencies, dark rates, overbias
      params, dead time, afterpulse, jitter, strict_mode, etc.), then
      ``SinglePhotonDetector.from_config_dict()`` validates ALL parameters
      in one atomic step via ``_validate_params()``.  No setattr overrides
      are needed.  The Geiger-mode guard lives solely in detectors.py.

    The returned overrides dict is now derived from ``detector.to_config_dict()``
    instead of manually tracking setattr calls, ensuring it always reflects
    the detector's actual runtime configuration.

    F-01: All detector parameter overrides are still logged at DEBUG level
    and recorded in the CSV output via the returned overrides dict.
    F-02/F-13: Removed all coverage-only throwaway detector constructions.
    """
    det_config_dict = _build_detector_config_dict(config)

    # ── Extract kwargs that from_config_dict() accepts via **extra_kwargs
    # but that are NOT dataclass fields (so they'd be rejected as unknown
    # dict keys).  Pop them out of the dict and pass them separately.
    _extra_kwargs = {}
    for _kw in ("flip_prob_override", "allow_xor_flip_formula"):
        if _kw in det_config_dict:
            _extra_kwargs[_kw] = det_config_dict.pop(_kw)

    detector = SinglePhotonDetector.from_config_dict(
        det_config_dict, **_extra_kwargs
    )

    # ── Derive overrides dict from the detector's actual config ──
    # This replaces the previous manual _overrides_applied dict that
    # tracked setattr calls.  Using to_config_dict() ensures the CSV
    # override columns always match what the detector is actually using,
    # even if from_config_dict() applied defaults different from the
    # input config.
    actual_config = detector.to_config_dict(include_experimental=True)
    input_config = det_config_dict

    _overrides_applied: Dict[str, Any] = {}
    override_keys = (
        "det_eff_d0", "det_eff_d1", "dark_rate", "dark_rate_d1",
        "bias_voltage", "breakdown_voltage", "temperature_k",
        "ref_temperature_k", "ref_bias_voltage",
    )
    for key in override_keys:
        input_val = input_config.get(key)
        actual_val = actual_config.get(key)
        # Record the value that the detector is actually using.
        # If the input and actual differ (e.g. due to default fallbacks
        # or internal normalization), both are logged for auditability.
        if actual_val is not None:
            _overrides_applied[key] = actual_val
            if input_val != actual_val:
                logger.debug(
                    "Detector parameter %s: input=%s, actual=%s (default/fallback applied).",
                    key, input_val, actual_val,
                )

    if _overrides_applied:
        logger.debug("Detector configuration (from to_config_dict): %s", _overrides_applied)

    return detector, _overrides_applied

def log_detector_details(detector, level: int = logging.DEBUG) -> None:
    """Log detector details using query methods from detectors.py.

    Mirrors log_source_details() which uses source object query methods.
    All values are read from the detector object (validated, frozen state)
    rather than the mutable config dict, ensuring logged values match
    what the detector actually uses during simulation.

    Uses ``to_config_dict()`` for a complete serializable snapshot and
    individual attribute reads for key physics parameters.
    """
    if not logger.isEnabledFor(level):
        return

    logger.log(level, f"Detector type: {type(detector).__name__}")
    logger.log(level, f"Detector detector_type: {detector.detector_type}")
    logger.log(level, f"det_eff_d0: {detector.det_eff_d0}")
    logger.log(level, f"det_eff_d1: {detector.det_eff_d1}")
    logger.log(level, f"dark_rate (D0): {detector.dark_rate} Hz")
    logger.log(level, f"dark_rate_d1: {detector.dark_rate_d1} Hz")
    logger.log(level, f"qber_intrinsic: {detector.qber_intrinsic}")
    logger.log(level, f"misalignment: {detector.misalignment}")
    logger.log(level, f"double_click_policy: {detector.double_click_policy}")
    logger.log(level, f"dead_time_ns: {detector.dead_time_ns}")
    logger.log(level, f"dead_time_model: {detector.dead_time_model}")
    logger.log(level, f"afterpulse_model: {detector.afterpulse_model}")
    logger.log(level, f"afterpulse_prob: {detector.afterpulse_prob}")
    logger.log(level, f"afterpulse_lifetime_ns: {detector.afterpulse_lifetime_ns}")
    logger.log(level, f"jitter_fwhm_ns: {detector.jitter_fwhm_ns}")
    logger.log(level, f"jitter_fwhm_ns_d0: {detector.jitter_fwhm_ns_d0}")
    logger.log(level, f"jitter_fwhm_ns_d1: {detector.jitter_fwhm_ns_d1}")
    logger.log(level, f"strict_mode: {detector.strict_mode}")
    logger.log(level, f"temperature_k: {detector.temperature_k}")
    logger.log(level, f"bias_voltage: {detector.bias_voltage}")
    logger.log(level, f"breakdown_voltage: {detector.breakdown_voltage}")
    logger.log(level, f"ref_temperature_k: {detector.ref_temperature_k}")
    logger.log(level, f"ref_bias_voltage: {detector.ref_bias_voltage}")
    logger.log(level, f"breakdown_temp_coeff_mv_per_k: {detector.breakdown_temp_coeff_mv_per_k}")
    logger.log(level, f"ap_reschedule_release: {detector.ap_reschedule_release}")
    logger.log(level, f"gated: {detector.gated}")
    logger.log(level, f"gate_width_ns: {detector.gate_width_ns}")
    logger.log(level, f"detector_topology: {detector.detector_topology}")

    # Per-detector overrides (if set)
    if detector.bias_voltage_d0 is not None:
        logger.log(level, f"bias_voltage_d0: {detector.bias_voltage_d0}")
    if detector.bias_voltage_d1 is not None:
        logger.log(level, f"bias_voltage_d1: {detector.bias_voltage_d1}")
    if detector.breakdown_voltage_d0 is not None:
        logger.log(level, f"breakdown_voltage_d0: {detector.breakdown_voltage_d0}")
    if detector.breakdown_voltage_d1 is not None:
        logger.log(level, f"breakdown_voltage_d1: {detector.breakdown_voltage_d1}")
    if detector.temperature_k_d0 is not None:
        logger.log(level, f"temperature_k_d0: {detector.temperature_k_d0}")
    if detector.temperature_k_d1 is not None:
        logger.log(level, f"temperature_k_d1: {detector.temperature_k_d1}")

    # Basis-dependent efficiency (if set)
    if detector.det_eff_d0_z is not None:
        logger.log(level, f"det_eff_d0_z: {detector.det_eff_d0_z}")
    if detector.det_eff_d0_x is not None:
        logger.log(level, f"det_eff_d0_x: {detector.det_eff_d0_x}")
    if detector.det_eff_d1_z is not None:
        logger.log(level, f"det_eff_d1_z: {detector.det_eff_d1_z}")
    if detector.det_eff_d1_x is not None:
        logger.log(level, f"det_eff_d1_x: {detector.det_eff_d1_x}")

    # Full config dict for complete audit trail
    logger.log(level, f"Detector config dict: {detector.to_config_dict(include_experimental=True)}")

def build_channel_from_distance(dist_km: float, config: Dict[str, Any]) -> FiberChannel:
    """Build a FiberChannel for the given distance.

    Uses ``build_attenuation_config_from_sim_config`` from channel.py
    to construct the AttenuationConfig, ensuring all channel parameter
    validation happens in channel.py.  Then builds a validated
    FiberChannel and logs its derived quantities.
    """
    attenuation = build_attenuation_config_from_sim_config(dist_km, config)
    ch = FiberChannel(attenuation, _construction_path="simulation")
    ch.validate()

    if ch.is_lossless:
        logger.debug("Channel is completely lossless.")

    logger.debug(
        f"Channel Tuple: {ch.as_tuple()}, "
        f"Linear Loss/km: {ch.single_km_power_loss_fraction():.6f}, "
        f"Config Dict: {ch.to_config_dict()}"
    )
    return ch


def build_protocol(config: Dict[str, Any], source) -> Protocol:
    pcfg = config["protocol_runtime"]
    protocol_class = pcfg["protocol_class"]
    if protocol_class == "BB84DecoyProtocol":
        return BB84DecoyProtocol(
            alice_z_basis_prob=float(pcfg["alice_z_basis_prob"]),
            bob_z_basis_prob=float(pcfg["bob_z_basis_prob"]),
            source=source, double_click_policy=pcfg["double_click_policy"],
        )
    elif protocol_class == "B92Protocol":
        return B92Protocol(
            bob_z_basis_prob=float(pcfg["bob_z_basis_prob"]),
            source=source,
        )
    elif protocol_class == "MDIQKDProtocol":
        return MDIQKDProtocol(z_basis_prob=float(pcfg["z_basis_prob"]), source=source)
    elif protocol_class == "RedundantTransmissionProtocol":
        return RedundantTransmissionProtocol(
            redundancy_M=int(pcfg["redundancy_M"]), source=source,
            use_entangling_decoder=bool(pcfg["use_entangling_decoder"]),
            use_entangling_encoder=bool(pcfg["use_entangling_encoder"]),
            codeword_mapping=pcfg["codeword_mapping"],
            damping_parameter=pcfg["damping_parameter"], rotation_angle=pcfg["rotation_angle"],
        )
    else:
        raise ConfigurationError(
            f"Unknown protocol_class: {protocol_class!r}.",
            context={
                "protocol_class": protocol_class,
                "valid_protocol_classes": [
                    "BB84DecoyProtocol",
                    "B92Protocol",
                    "MDIQKDProtocol",
                    "RedundantTransmissionProtocol",
                ],
            },
        )

def get_pulse_indices_from_prepared_states(prepared_states) -> np.ndarray:
    if hasattr(prepared_states, "alice_pulse_type_indices"):
        return np.asarray(prepared_states.alice_pulse_type_indices, dtype=np.int64)
    raise QKDSimulationError(
        "Prepared states object does not expose alice_pulse_type_indices.",
        context={"prepared_states_type": type(prepared_states).__name__},
    )

def generate_photons_for_prepared_states(source, prepared_states, rng) -> np.ndarray:
    pulse_indices = get_pulse_indices_from_prepared_states(prepared_states)
    num_samples = len(pulse_indices)
    photons = source.generate_photons(alice_pulse_indices=pulse_indices, rng=rng, num_samples=num_samples)
    return np.asarray(photons, dtype=np.int64)

def infer_ideal_outcomes_d0(prepared_states) -> np.ndarray:
    if isinstance(prepared_states, BB84PreparedStates):
        return (prepared_states.alice_bits == 0)
    if isinstance(prepared_states, B92PreparedStates):
        return (prepared_states.alice_bits == 0)
    if isinstance(prepared_states, MDIPreparedStates):
        parity = np.bitwise_xor(prepared_states.alice_bits, prepared_states.bob_bits)
        return (parity == 0)
    raise QKDSimulationError(
        "Unsupported prepared states type for ideal outcome inference.",
        context={"prepared_states_type": type(prepared_states).__name__},
    )

def extract_detection_arrays(det_result, sample_size: int):
    """Pull the two click arrays out of a ``DetectionResult``."""
    if not isinstance(det_result, DetectionResult):
        if isinstance(det_result, tuple) and len(det_result) >= 2:
            d0, d1 = det_result[0], det_result[1]
        else:
            raise QKDSimulationError(
                "Could not extract detector click arrays from DetectionResult.",
                context={"det_result_type": type(det_result).__name__, "sample_size": sample_size},
            )
    else:
        d0 = det_result.click0
        d1 = det_result.click1

    arr0 = np.asarray(d0).reshape(sample_size)
    arr1 = np.asarray(d1).reshape(sample_size)
    if np.issubdtype(arr0.dtype, np.integer):
        return arr0, arr1
    return arr0.astype(bool), arr1.astype(bool)

def detector_result_to_protocol_detection_results(det_result, num_pulses: int) -> DetectionResults:
    click0, click1 = extract_detection_arrays(det_result, num_pulses)
    metadata: Dict[str, Any] = {}

    if det_result.diagnostics is not None:
        diag = det_result.diagnostics
        metadata.update(diag.to_metadata_dict())

    # ``DetectionResult.state_snapshot`` (renamed from ``final_state`` in
    # the detectors module rewrite) carries the JSON state string.  We
    # keep the CSV column name ``final_state_snippet`` for backward
    # compatibility with downstream analysis scripts.
    state_snapshot = getattr(det_result, "state_snapshot", None)
    if state_snapshot is None:
        # Backward-compat: older ``DetectionResult`` versions exposed
        # the same data under the ``final_state`` attribute.
        state_snapshot = getattr(det_result, "final_state", None)
    if state_snapshot is not None:
        metadata["final_state_snippet"] = state_snapshot[:50]

    return DetectionResults(num_pulses=num_pulses, click0=click0, click1=click1, metadata=metadata if metadata else None)

def _basis_is_z_mask(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype == np.bool_:
        return arr.astype(bool)
    if np.issubdtype(arr.dtype, np.integer):
        return arr == 0
    out = np.zeros(arr.shape, dtype=bool)
    flat = arr.ravel()
    for i, v in enumerate(flat):
        if isinstance(v, str):
            s = v.strip().upper()
        else:
            name = getattr(v, "name", None)
            s = str(name).strip().upper() if name is not None else str(v).strip().upper()
        out.ravel()[i] = s in {"Z", "Z_BASIS", "RECTILINEAR", "STANDARD", "0"}
    return out

def _get_first_existing_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None

def tally_counts_from_sifting_results(prepared_states, sifting_results: SiftingResults) -> Dict[str, TallyCounts]:
    result: Dict[str, TallyCounts] = {}
    pulse_indices = np.asarray(prepared_states.alice_pulse_type_indices, dtype=np.int64)
    unique_indices = np.unique(pulse_indices)
    alice_basis_values = _get_first_existing_attr(prepared_states, ["alice_bases", "alice_basis", "bases", "basis", "alice_basis_choices", "basis_choices"])
    bob_basis_values = _get_first_existing_attr(sifting_results, ["bob_bases", "bob_basis", "bob_basis_choices", "measurement_bases", "measurement_basis"])
    alice_z_mask = _basis_is_z_mask(alice_basis_values) if alice_basis_values is not None else None
    alice_x_mask = ~alice_z_mask if alice_z_mask is not None else None
    bob_z_mask = _basis_is_z_mask(bob_basis_values) if bob_basis_values is not None else None
    bob_x_mask = ~bob_z_mask if bob_z_mask is not None else None
    global_sifted_mask = np.asarray(sifting_results.sifted_mask, dtype=bool)
    global_error_mask = np.asarray(sifting_results.error_mask, dtype=bool)
    for idx in unique_indices:
        sent_mask = pulse_indices == idx
        sifted_mask = sent_mask & global_sifted_mask
        err_mask = sent_mask & global_sifted_mask & global_error_mask
        sent = int(np.count_nonzero(sent_mask))
        sifted = int(np.count_nonzero(sifted_mask))
        errors = int(np.count_nonzero(err_mask))
        if alice_z_mask is not None:
            sent_z, sent_x = int(np.count_nonzero(sent_mask & alice_z_mask)), int(np.count_nonzero(sent_mask & alice_x_mask))
        else:
            sent_z, sent_x = int(round(sent * 0.5)), sent - int(round(sent * 0.5))
        if alice_z_mask is not None and bob_z_mask is not None:
            sifted_z, sifted_x = int(np.count_nonzero(sifted_mask & alice_z_mask & bob_z_mask)), int(np.count_nonzero(sifted_mask & alice_x_mask & bob_x_mask))
            errors_sifted_z, errors_sifted_x = int(np.count_nonzero(err_mask & alice_z_mask & bob_z_mask)), int(np.count_nonzero(err_mask & alice_x_mask & bob_x_mask))
        elif alice_z_mask is not None:
            sifted_z, sifted_x = int(np.count_nonzero(sifted_mask & alice_z_mask)), int(np.count_nonzero(sifted_mask & alice_x_mask))
            errors_sifted_z, errors_sifted_x = int(np.count_nonzero(err_mask & alice_z_mask)), int(np.count_nonzero(err_mask & alice_x_mask))
        else:
            sifted_z, sifted_x = int(round(sifted * 0.5)), sifted - int(round(sifted * 0.5))
            errors_sifted_z, errors_sifted_x = int(round(errors * 0.5)), errors - int(round(errors * 0.5))
        result[str(idx)] = TallyCounts(
            sent=sent, sifted=sifted, errors_sifted=errors, double_clicks_discarded=0,
            sent_z=sent_z, sent_x=sent_x, sifted_z=sifted_z, sifted_x=sifted_x,
            errors_sifted_z=errors_sifted_z, errors_sifted_x=errors_sifted_x,
        )
    return result

def summarize_tallies(stats_map: Dict[str, TallyCounts]) -> TallyCounts:
    total = TallyCounts()
    for tc in stats_map.values():
        total = total.merged(tc)
    return total

def roundtrip_tallycounts(stats_map: Dict[str, TallyCounts]) -> Dict[str, TallyCounts]:
    return {k: TallyCounts.from_dict(v.to_dict()) for k, v in stats_map.items()}


# ============== VACUUM ERROR-RATE CORRECTION ==============
#
# Background:
#   Lim et al. 2014 (PhysRevA.89.022307) Appendix B explicitly assumes that
#   "vacuum contributions contain zero information about the chosen bit values
#   and the bits are uniformly distributed."  This is what justifies giving
#   the vacuum events s_{X,0} full entropy credit in the key-length formula
#   (Eq. 1) without any h(.) subtraction.
#
#   Physically, vacuum detections are dark counts: the detector fires
#   spontaneously and the bit value Bob assigns is uniformly random in
#   {0,1}.  Therefore the *measured* QBER on vacuum pulses must be ~50%,
#   not the detector's intrinsic QBER (qber_intrinsic ~ 1%).
#
# Bug:
#   The qkd.detectors package applies the intrinsic-QBER / misalignment
#   noise model uniformly to every click, including dark-count clicks.
#   This produces qber_vacuum ~ 1% at short distances, which violates
#   Lim2014 Appendix B's assumption.  When fed into Eq. (4) for v_{Z,1}^U,
#   the under-estimated m_{Z,mu3} lets the proof subtract too little of
#   the dark-count contribution, which inflates v_{Z,1}^U, inflates the
#   phase-error bound phi_X^U via Eq. (5), and ultimately drives the
#   key length negative (LPFailureError -> ZERO_KEY) at 0/10 km.
#
# Fix (workaround at the stats-map boundary):
#   Since the qkd.detectors source is not part of this repository, we
#   correct the vacuum-pulse error count at the boundary between the
#   simulator and the Lim2014 proof.  For each vacuum pulse, we replace
#   errors_sifted_{z,x} with round(0.5 * sifted_{z,x}), which is the
#   value a physically-correct dark-count model would produce in
#   expectation.  This makes the measured qber_vacuum consistent with
#   the paper's Appendix B assumption, restoring the proof's internal
#   consistency.
#
#   NOTE: This is a workaround.  The proper fix lives inside
#   qkd.detectors, where dark-count clicks should be assigned uniformly
#   random bits instead of inheriting the intrinsic-QBER model.  When
#   that fix is available upstream, this correction becomes a no-op for
#   correct simulations and a defensive guard against regressions.






# NOTE (2024): The signal/decoy QBER bug described below has been FIXED
# upstream in qkd/detector.py. Measured qber_signal ≈ 1.02% now matches
# the configured qber_intrinsic. No correction is needed.
#
# Original bug description (now fixed):
#   qkd.detectors previously produced qber_signal ≈ 0.235 and
#   qber_decoy ≈ 0.41 due to the noise-model bug. This is no longer
#   observed in simulation output.

def roundtrip_pulse_configs(source) -> List[Dict[str, Any]]:
    items = []
    for pc in source.pulse_configs:
        if hasattr(pc, "to_dict"):
            items.append(pc.to_dict())
        else:
            items.append({"name": pc.name, "mean_photon_number": pc.mean_photon_number, "probability": pc.probability})
    return items

def build_simulation_results(protocol_params, config, stats_map, secure_key, raw_sifted, sim_time, status_text, metadata, source):
    lp_diag = {
        "preferred_lp_solver": config["source"]["preferred_lp_solver"],
        "source_class": config["source"]["source_class"],
        "decoder_architecture": config["source"]["expected_decoder"],
        "protocol_class": config["protocol_runtime"]["protocol_class"],
    }
    certificate = build_security_certificate(config, lp_solver_diagnostics=lp_diag)
    status_enum = SimulationStatus.OK if status_text == "OK" else SimulationStatus.FAILED
    results = SimulationResults(
        params=protocol_params, metadata=metadata, security_certificate=certificate,
        decoy_estimates={"pulse_configs_roundtrip": roundtrip_pulse_configs(source), "intensity_config": protocol_params.intensities.to_dict()},
        secure_key_length=int(secure_key), raw_sifted_key_length=int(raw_sifted),
        simulation_time_seconds=float(sim_time), status=status_enum,
        tally_stats=roundtrip_tallycounts(stats_map),
        weak_gllp_rate=None, tagged_fraction_bound=None,
        beta_y1_deviation=None, beta_e1_deviation=None,
        success_probability=1.0 if secure_key > 0 else 0.0,
    )
    return results


# ============== WORKER STATE ==============
# F-03: Worker mutable state is encapsulated in a single _WorkerState instance
# instead of scattered module-level globals.  Each worker process gets its
# own copy via the initializer; the instance is NOT shared across processes.

class _WorkerState:
    """Per-worker mutable state.  Set by init_worker(), read by run_single_simulation().

    Not thread-safe — each worker process owns its own instance.
    F-14: source_meta_cache avoids re-serialising the same source metadata
    for every distance/pulse-count row when the combo hasn't changed.
    """
    __slots__ = (
        "base_config", "combo_map", "last_combo_idx",
        "source", "detector", "protocol", "config",
        "source_meta_cache", "detector_overrides",
        # Auto-optimize state: per-(combo, distance) optimizer cache.
        # Avoids re-running SLSQP when only `apply_noise` flips between
        # the noise=False and noise=True rows at the same (combo, dist).
        "auto_optimize_pulses", "last_optimized_dist", "optimized_params",
        # DWDM mode: when True, run_single_simulation calls
        # load_lim2014_dwdm_params instead of load_lim2014_dedicated_params.
        "dwdm",
        "wdm_channel_count",  # number of classical WDM channels
        "wdm_channel_power_dbm",  # None = paper-faithful -34 dBm received
        "params",  # cached QKDParams for get_params_metadata_row on error path
        "wdm_raman_coefficient",
    )

    def __init__(self) -> None:
        self.base_config: Optional[Dict[str, Any]] = None
        self.combo_map: Optional[Dict[int, Dict[str, Any]]] = None
        self.last_combo_idx: int = -1
        self.source = None
        self.detector = None
        self.protocol = None
        self.config: Optional[Dict[str, Any]] = None
        self.source_meta_cache: Dict[int, Dict[str, Any]] = {}
        self.detector_overrides: Dict[str, Any] = {}
        self.auto_optimize_pulses: bool = False
        self.last_optimized_dist: Optional[Tuple[int, float]] = None
        self.optimized_params: Optional[Tuple[float, float, float, float, float]] = None
        self.dwdm: bool = False
        self.wdm_channel_count: int = 4
        self.wdm_channel_power_dbm = None  # None = paper-faithful (-34 dBm received)


_ws = _WorkerState()

# Detector metadata keys are derived from ``DetectionDiagnostics`` so that
# new diagnostic fields are picked up automatically and the CSV header never
# drifts from the dataclass.  ``final_state_snippet`` is appended because it
# is sourced from ``DetectionResult.state_snapshot`` (a JSON state string,
# renamed from ``final_state`` in the detectors module rewrite), not from
# ``DetectionDiagnostics``.
DETECTOR_METADATA_KEYS = list(DetectionDiagnostics.metadata_keys()) + [
    "final_state_snippet",
]

# F-01: Additional CSV columns that record detector parameter overrides
# applied by build_detector(), making previously-silent modifications
# auditable in the output data.
DETECTOR_OVERRIDE_KEYS = [
    "det_override_det_eff_d0",
    "det_override_det_eff_d1",
    "det_override_dark_rate",
    "det_override_dark_rate_d1",
    "det_override_bias_voltage",
    "det_override_breakdown_voltage",
    "det_override_temperature_k",
    "det_override_ref_temperature_k",
    "det_override_ref_bias_voltage",
]

# DWDM / channel-filter metadata columns.  These live on the QKDParams object
# (not on the source), so they require a separate helper get_params_metadata_row()
# to be extracted into the CSV row.  Without these columns, a --dwdm run is
# indistinguishable from a dedicated run in the CSV output.
PARAMS_METADATA_KEYS = [
    "params_wdm_channel_count",
    "params_wdm_channel_power_dbm",
    "params_wdm_raman_coefficient",
    "params_filter_model",
    "params_filter_awg_loss_db",
    "params_awg_isolation_db",
    "params_channel_total_loss_db",
    "params_raman_dark_rate_hz",
]

def init_worker(base_config, combo_map, auto_optimize_pulses=False, dwdm=False,
                  wdm_channel_count=4, wdm_channel_power_dbm=None, wdm_raman_coefficient: float = None):
    """Initialize workers with the base config and the map of sweep combinations.

    The optional ``auto_optimize_pulses`` flag enables per-distance pulse
    parameter optimization (mu, nu, p_s, p_d, p_v) inside ``run_single_simulation``.
    Each (combo_idx, distance) pair gets its own optimized parameters, cached
    on the worker state so that the noise=False and noise=True rows at the
    same (combo, dist) share the same optimized source.
    """
    # Configure logging in each worker process so that ERROR-level
    # messages from failed simulation points are visible in the
    # terminal.  The root logger configuration set in
    # run_and_save_csv() is only effective in the main process;
    # worker processes (spawned via multiprocessing) start with
    # the default WARNING-level root logger, which silently
    # discards ERROR-level logs from _log_qkd_exception().
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    _ws.base_config = base_config
    _ws.combo_map = combo_map
    _ws.last_combo_idx = -1
    _ws.source_meta_cache = {}
    _ws.detector_overrides = {}
    _ws.auto_optimize_pulses = bool(auto_optimize_pulses)
    _ws.last_optimized_dist = None
    _ws.optimized_params = None
    _ws.params = None  # cached QKDParams for get_params_metadata_row on error path
    _ws.dwdm = bool(dwdm)
    # FIX-WS-WDM-V2: exact _ws.wdm_* assignments
    _ws.wdm_channel_count = int(wdm_channel_count) if wdm_channel_count is not None else None
    _ws.wdm_channel_power_dbm = float(wdm_channel_power_dbm) if wdm_channel_power_dbm is not None else None
    _ws.wdm_raman_coefficient = float(wdm_raman_coefficient) if wdm_raman_coefficient is not None else None
    # Option H: removed duplicate raw assignments that were overwriting the
    # type-coerced None-safe versions above.

def get_source_metadata_row(source) -> Dict[str, Any]:
    """Extract source metadata using source object properties (modular coupling).

    Refactored to read from the source object's validated, frozen properties
    (e.g., ``source.statistics_type``, ``source.modulation_index``) instead of
    the ``to_dict()`` serialization path.  This ensures parameter consistency:
    the metadata always reflects the source's internal state, not a potentially
    stale or lossy dict round-trip.

    F-02: Removed calculate_effective_mus() coverage probe that created a
    throwaway RNG on every call.  Effective μ values can be computed
    separately in analysis if needed.
    F-14: This function is now cached per combo_idx via _ws.source_meta_cache.
    """
    pulse_details = {}
    for name in source.pulse_names():
        pulse_details[f"pulse_{name}_mu"] = source.get_mean_photon_number_by_name(name)
        pulse_details[f"pulse_{name}_prob"] = source.get_pulse_probability_by_name(name)

    mzm_status = "none"
    if source.mzm is not None:
        er = getattr(source.mzm, "extinction_ratio_db", None)
        mzm_status = f"er_{er}db" if er is not None else "active"

    electrical_noise_std = 0.0
    if source.electrical_noise is not None:
        electrical_noise_std = source.electrical_noise.total_voltage_std()

    security_enabled = False
    security_total_sent = 0
    if source.security_metadata is not None:
        security_enabled = True
        security_total_sent = getattr(source.security_metadata, "total_sent", 0)

    return {
        "source_type": type(source).__name__,
        "uses_density_matrices": isinstance(source, DensityMatrixSource) and bool(source.density_matrices),
        "ideal_emission_probability": source.ideal_emission_probability,
        "statistics_type": source.statistics_type.value,
        "error_model": source.error_model.value,
        "modulation_index": source.modulation_index,
        "intensity_jitter": source.intensity_jitter,
        "N_channels": source.N_channels,
        "signal_pulse_index": source.get_pulse_index_by_name("signal"),
        "use_small_angle_approximation": source.use_small_angle_approximation,
        "use_linear_modulation_approximation": source.use_linear_modulation_approximation,
        # Access config.is_bidirectional directly to avoid the deprecated
        # property warning on OpticalSource.is_bidirectional.
        "is_bidirectional": source.config.is_bidirectional,
        "mzm_status": mzm_status,
        "electrical_noise_std_v": electrical_noise_std,
        **pulse_details,
        "security_metadata_enabled": security_enabled,
        "security_metadata_total_sent": security_total_sent,
    }

def _compute_raman_dark_rate_hz(params) -> float:
    """Compute the Raman-noise-equivalent dark rate (Hz) for a QKDParams object.

    This is the active-path port of SimulationNoiseMixin._calculate_nonlinear_noise
    (qkd/simulation_noise.py lines 35-71).  That mixin is composed into QKDSystem,
    but main_optimized.py does NOT use QKDSystem -- it calls detector.simulate_detection()
    directly.  Without this port, wdm_raman_coefficient / wdm_channel_count /
    wdm_channel_power_dbm are never consumed by the active pipeline.

    Physics (per Lim2014 Section IV, Ref. [35]):
        p_launch    = 10^(wdm_channel_power_dbm/10) * 1e-3   # Watts
        alpha       = fiber_loss_db_km * 0.23026              # 1/km (dB/km -> Np/km, *ln(10)/10)
        l_eff       = (1 - exp(-alpha * L)) / alpha           # nonlinear effective length
        p_raman     = p_launch * (N-1) * wdm_raman_coeff * l_eff
        photon_E    = h*c / lambda
        rate_hz     = p_raman / photon_E                      # photons/sec
        # Per-detector dark-rate equivalent: rate * gate_width * det_eff converts
        # to per-gate click probability; dividing by gate_width returns Hz so it
        # can be added to detector.dark_rate.  The det_eff factor is folded in
        # because detector.dark_rate is the *post-detection-efficiency* rate.
        y0_raman_hz = rate_hz * det_eff_d0

    Returns 0.0 when wdm_channel_count <= 1 or any required field is missing.
    """
    import math
    import scipy.constants as _const

    if params is None:
        return 0.0
    wdm_count = getattr(params, "wdm_channel_count", 1)
    if wdm_count is None or wdm_count <= 1:
        return 0.0

    wdm_power_dbm = float(getattr(params, "wdm_channel_power_dbm", 0.0) or 0.0)
    p_launch = (10.0 ** (wdm_power_dbm / 10.0)) * 1e-3  # Watts

    ch = getattr(params, "channel", None)
    if ch is None:
        return 0.0
    # Option J: prefer params.fiber_loss_db_km (pure fiber attenuation) over
    # channel.fiber_loss_db_km.  In DWDM mode, load_lim2014_dwdm_params rebuilds
    # the channel via FiberChannel.from_total_loss(distance, fiber_loss*L +
    # awg_loss), which derives alpha = total_loss_db / distance_km -- mixing
    # fiber + AWG filter loss.  Using that contaminated alpha in l_eff
    # underestimates Raman by 9-19% depending on distance.
    fiber_loss_db_km = float(getattr(params, "fiber_loss_db_km",
                                     getattr(ch, "fiber_loss_db_km", 0.2)))
    distance_km = float(getattr(ch, "distance_km", 0.0))
    alpha = fiber_loss_db_km * 0.23026  # Np/km
    if alpha > 1e-12:
        l_eff = (1.0 - math.exp(-alpha * distance_km)) / alpha
    else:
        l_eff = distance_km

    wdm_raman_coeff = float(getattr(params, "wdm_raman_coefficient", 0.0))
    p_raman_broadband = p_launch * (wdm_count - 1) * wdm_raman_coeff * l_eff

    # AWG filter rejection: broadband Raman is spread over ~12 THz
    # (S+C+L bands) but only the fraction within the QKD channel's
    # AWG passband (~50 GHz) reaches the detector.  Typical AWG
    # adjacent-channel isolation is 25-30 dB.  Default: 25 dB.
    _awg_isolation_db = float(getattr(params, "awg_isolation_db", 25.0))
    _raman_filter_factor = 10.0 ** (-_awg_isolation_db / 10.0)
    p_raman = p_raman_broadband * _raman_filter_factor

    lam_nm = float(getattr(params, "carrier_wavelength_nm", 1550.0))
    if lam_nm <= 0:
        return 0.0
    photon_energy = (_const.h * _const.c) / (lam_nm * 1e-9)
    if photon_energy <= 0:
        return 0.0

    rate_hz = p_raman / photon_energy  # photons/sec

    det = getattr(params, "detector", None)
    det_eff_d0 = float(getattr(det, "det_eff_d0", 1.0)) if det is not None else 1.0

    # rate_hz is the photon-arrival rate at the detector.
    # detector.dark_rate is the post-detection click rate (already includes det_eff).
    # Multiply by det_eff_d0 so the bumped dark_rate yields the correct extra
    # click probability per gate.
    return rate_hz * det_eff_d0

def _load_lim2014_dwdm_params_safe(distance_km, num_bits, fiber_loss_db_km,
                                    wdm_channel_count, wdm_channel_power_dbm):
    """Load LIM2014 DWDM params, working around FiberChannel.from_total_loss
    rejecting non-zero loss at distance=0."""
    try:
        return load_lim2014_dwdm_params(
            distance_km=distance_km, num_bits=num_bits,
            fiber_loss_db_km=fiber_loss_db_km,
            wdm_channel_count=wdm_channel_count,
            wdm_channel_power_dbm=wdm_channel_power_dbm,
        )
    except (ValueError, Exception) as _exc_dwdm:
        if distance_km != 0.0:
            raise
        import dataclasses as _dc_dwdm
        import logging as _log_dwdm
        _log_dwdm.getLogger('main_optimized').debug(
            f'load_lim2014_dwdm_params failed at distance=0 '
            f'({type(_exc_dwdm).__name__}: {_exc_dwdm}); '
            f'falling back to dedicated-fiber factory + DWDM field patch'
        )
        params = load_lim2014_dedicated_params(
            distance_km=0.0, num_bits=num_bits,
            fiber_loss_db_km=fiber_loss_db_km,
        )
        # Patch DWDM-specific scalar fields.
        _awg_db = 3.0  # default AWG filter loss
        if hasattr(params, 'wdm_channel_count'):
            params = _dc_dwdm.replace(params, wdm_channel_count=wdm_channel_count)
        if hasattr(params, 'wdm_channel_power_dbm'):
            # load_lim2014_dwdm_params computes wdm_power from distance.
            # At d=0, use a reasonable default (-30 dBm received).
            _awg_db_fb = float(getattr(params, 'filter_awg_loss_db', 3.0))
            _wdm_pwr = wdm_channel_power_dbm if wdm_channel_power_dbm is not None else -34.0 + _awg_db_fb
            params = _dc_dwdm.replace(params, wdm_channel_power_dbm=_wdm_pwr)
        if hasattr(params, 'wdm_raman_coefficient'):
            params = _dc_dwdm.replace(params, wdm_raman_coefficient=1e-9)
        if hasattr(params, 'filter_awg_loss_db'):
            params = _dc_dwdm.replace(params, filter_awg_loss_db=_awg_db)
        # Patch channel with AWG loss.  At d=0, the dedicated factory
        # creates a channel with total_loss=0.  We need total_loss=3 dB
        # (AWG only, no fiber).  Build a new channel with the AWG loss
        # included via from_config_dict which allows fixed losses.
        # At d=0, we cannot build a FiberChannel with non-zero loss.
        # Leave params.channel as-is (total_loss_db=0). The ch_params
        # patch in the simulation loop will correct the effective loss
        # for the simulation. The proof will also get corrected loss    
        # via the ch_params patch or the DWDM fallback patches.
        return params

def get_params_metadata_row(params) -> Dict[str, Any]:
    """Extract DWDM / channel-filter metadata from a QKDParams object.

    The source metadata helper (get_source_metadata_row) only sees the source
    object, which does NOT carry wdm_channel_count / filter_model / etc.
    Those fields live on QKDParams.  This helper closes the verification gap
    so that a --dwdm run is auditable from the CSV alone.

    Returns None for every key when ``params`` is None or lacks the field,
    so the error path (where params may not have been built) still produces
    a row with the correct schema.
    """
    if params is None:
        return {k: None for k in PARAMS_METADATA_KEYS}

    def _get(name, default=None):
        return getattr(params, name, default)

    channel_total_loss_db = None
    ch = _get("channel", None)
    if ch is not None and hasattr(ch, "total_loss_db"):
        channel_total_loss_db = float(ch.total_loss_db)

    return {
        "params_wdm_channel_count": _get("wdm_channel_count", 1),
        "params_wdm_channel_power_dbm": _get("wdm_channel_power_dbm", 0.0),
        "params_wdm_raman_coefficient": _get("wdm_raman_coefficient", 0.0),
        "params_filter_model": _get("filter_model", "ideal"),
        "params_filter_awg_loss_db": _get("filter_awg_loss_db", 0.0),
        "params_awg_isolation_db": _get("awg_isolation_db", 25.0),
        "params_channel_total_loss_db": channel_total_loss_db,
        "params_raman_dark_rate_hz": _compute_raman_dark_rate_hz(params),
    }

def _build_detector_override_row(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """F-01: Build detector override CSV columns from the stored overrides dict."""
    return {
        f"det_override_{k}": overrides.get(k)
        for k in ("det_eff_d0", "det_eff_d1", "dark_rate", "dark_rate_d1",
                   "bias_voltage", "breakdown_voltage", "temperature_k",
                   "ref_temperature_k", "ref_bias_voltage")
    }


# ============== ANALYTICAL P(n) LOOKUP TABLE BUILDER ==============
#
# Uses photon_number_distribution / poisson_pn_array / thermal_pn_array from
# sources.py to build a pre-computed P(n) lookup table for all pulse types.
# This table is a raw NumPy array (shape: [n_pulse_types, PN_TRUNCATION_DIM])
# that can be consumed by Numba-compiled loops or vectorized gain-curve
# pre-computation without any Python-object overhead.
#
# This directly addresses the "expose raw NumPy arrays" refactoring criterion:
# source physics (P(n) computation) stays in sources.py, while main_optimized
# receives a ready-to-use array for tight-loop performance.

def build_pn_lookup_table(
    source: OpticalSource,
    max_n: int = PN_TRUNCATION_DIM,
) -> np.ndarray:
    """Build a pre-computed P(n) lookup table for all pulse types.

    Returns a float64 array of shape (n_pulse_types, max_n) where element
    [i, n] holds P(n) for pulse type i.  Uses
    ``photon_number_distribution`` from sources.py (single source-of-truth
    for P(n) computation) instead of duplicating Poisson/thermal formulas.

    Parameters
    ----------
    source : OpticalSource
        A fully constructed and validated source object.
    max_n : int, default PN_TRUNCATION_DIM
        Truncation dimension (number of photon-number bins).

    Returns
    -------
    np.ndarray
        Probability table of shape (n_pulse_types, max_n), dtype float64.
        Each row sums to approximately 1 (truncation error < 1e-18 for
        mu < 1).
    """
    pulse_names = source.pulse_names()
    n_pulses = len(pulse_names)
    table = np.zeros((n_pulses, max_n), dtype=np.float64)
    for i, name in enumerate(pulse_names):
        mu = source.get_mean_photon_number_by_name(name)
        table[i, :] = photon_number_distribution(
            mu, source.statistics_type, max_n=max_n
        )
    return table


# ============== ROGERS et al. (2007) ANALYTIC CROSS-CHECK ==============
#
# The patched qkd.detectors module exposes closed-form expressions for the
# sifted-bit rate (SBR) of a BB84 system under dead-time effects, derived
# in Rogers, Bienfang, Nakassis, Xu & Clark (2007).  The relevant formula
# is Eq. (15):
#
#     SBR = rho_TX * 8 * p * P_00(p, k) * S(p, k)
#
# where p = L/8 is the per-detector per-cycle click probability, k is the
# number of transmission periods per dead time (k = rho_TX * tau), P_00 is
# the steady-state probability that both detectors in a basis are alive
# (Eq. 8), and S(p, k) is the sifting likelihood (Eq. 9).  We compute the
# analytic SBR for every simulation point and emit it as a CSV column so
# the user can directly compare the Monte-Carlo secure_key_rate against
# the paper's analytic curve (Figs. 3-5).
#
# Caveats:
# - The paper's ``L`` is the probability that a photon emitted by Alice is
#   detected at Bob.  We now compute it via ``link_efficiency`` from
#   ``qkd.channel``, which centralises the eta_channel * eta_detector
#   composition.  Previously this was an inline multiplication that could
#   diverge from the detector's actual efficiency if config-dict reads
#   and detector object attributes differed.
# - The analytic model assumes BB84 with random bases and a single
#   basis-pair of detectors.  Decoy-state / MDI / redundant-transmission
#   protocols will deviate.
# - The analytic model is noiseless in its basic form (eps=0); we expose
#   ``eps`` as a future extension hook but do not currently populate it
#   from the config.

def _compute_rogers_analytic_sbr(
    *,
    pulse_period_ns: float,
    dead_time_ns: float,
    channel_transmittance: float,
    det_eff_d0: float,
    coupling_efficiency: float = 1.0,
    eps: float = 0.0,
) -> Optional[float]:
    """Compute the Rogers et al. (2007) analytic sifted-bit rate (Eq. 15).

    Uses ``link_efficiency`` from ``qkd.channel`` to compute the overall
    detection probability ``L = eta_ch * eta_det * eta_coup``, ensuring
    the link-budget composition is consistent with the decoy-state gain
    formula and any other code that computes overall detection
    probability.

    Returns ``None`` when the inputs are out of the analytic model's
    domain (zero dead time, zero transmittance, etc.), so the caller can
    emit a null CSV cell rather than a misleading 0.0.
    """
    if dead_time_ns <= 0.0 or pulse_period_ns <= 0.0:
        return None

    # Compute link efficiency via channel.py helper, ensuring
    # parameter consistency with the rest of the simulation.
    L = link_efficiency(channel_transmittance, det_eff_d0, coupling_efficiency)

    if L <= 0.0:
        return None
    L = min(L, 1.0)

    rho_tx = 1e9 / float(pulse_period_ns)              # Hz
    tau_s = float(dead_time_ns) * 1e-9                  # s

    try:
        return float(rogers_sifted_bit_rate(
            rho_tx=rho_tx, L=L, tau=tau_s, eps=eps,
        ))
    except (ValueError, OverflowError, ArithmeticError) as exc:
        logger.debug(
            "Rogers analytic SBR computation failed: %s. "
            "(rho_tx=%g, L=%g, tau=%g, eps=%g)",
            exc, rho_tx, L, tau_s, eps,
        )
        return None


def _balance_decoy_probabilities(
    overrides: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Auto-balance decoy pulse probabilities during sweep override processing.

    When a sweep overrides one pulse probability (signal or decoy) but not
    the other, this function computes the missing probability so that all
    probabilities sum to 1.0.

    NOTE: This runs BEFORE the source is constructed, so it reads from the
    mutable config dict rather than using source.pulse_probabilities().
    Once the source object exists, downstream code should prefer
    source.intensity_config and source.pulse_probabilities() for validated,
    role-based pulse data.

    For the standard 3-pulse BB84 decoy-state protocol:
        p_signal + p_decoy + p_vacuum = 1.0
    If p_signal is swept, p_decoy is computed as 1 - p_signal - p_vacuum.
    If p_decoy is swept, p_signal is computed as 1 - p_decoy - p_vacuum.

    F-11: Validates that balanced probabilities stay in [0, 1].
    """
    sig_p = overrides.get("source.pulses.signal.prob")
    dec_p = overrides.get("source.pulses.decoy.prob")
    vac_p = config["source"]["pulses"]["vacuum"]["prob"]

    if sig_p is not None and dec_p is None:
        balanced_dec = round(1.0 - sig_p - vac_p, 9)
        if balanced_dec < 0:
            raise ConfigurationError(
                f"Probability balancing produced negative decoy probability ({balanced_dec}). "
                f"signal.prob={sig_p}, vacuum.prob={vac_p}.",
                context={"signal_prob": sig_p, "vacuum_prob": vac_p, "decoy_prob": balanced_dec},
            )
        overrides["source.pulses.decoy.prob"] = balanced_dec
    elif dec_p is not None and sig_p is None:
        balanced_sig = round(1.0 - dec_p - vac_p, 9)
        if balanced_sig < 0:
            raise ConfigurationError(
                f"Probability balancing produced negative signal probability ({balanced_sig}). "
                f"decoy.prob={dec_p}, vacuum.prob={vac_p}.",
                context={"decoy_prob": dec_p, "vacuum_prob": vac_p, "signal_prob": balanced_sig},
            )
        overrides["source.pulses.signal.prob"] = balanced_sig


# ==================== PER-DISTANCE PULSE OPTIMIZER ====================
# The proof-aware SLSQP optimizer now lives in qkd.proofs.optimization
# (single source of truth).  This thin wrapper preserves the internal
# function name so existing call sites (run_single_simulation) don't need
# to change.  New code should import optimize_pulses_for_distance directly
# from qkd.proofs.optimization.


def _optimize_pulses_for_distance(config: Dict[str, Any], dist_km: float,
                                  total_pulses: int, proof_name: str
                                  ) -> Tuple[float, float, float, float, float]:
    """Thin wrapper around qkd.proofs.optimization.optimize_pulses_for_distance.

    Delegates to the shared optimizer in qkd.proofs.optimization so that
    main_optimized.py, lp_validation.py, and qkd_ui.py all use the same
    proof-aware SLSQP implementation.
    """
    return optimize_pulses_for_distance(config, dist_km, total_pulses, proof_name)


def run_single_simulation(args_tuple) -> Dict[str, Any]:
    """Worker function - rebuilds hardware only when sweep parameters change.

    F-03: Uses _WorkerState instead of module-level globals.
    F-05: Uses SeedSequence.spawn for cryptographically-sound RNG derivation.
    """
    dist, total_pulses, apply_noise, worker_seed, combo_idx = args_tuple

    # If this is a new combination of sweep parameters, rebuild the physical layer
    if combo_idx != _ws.last_combo_idx:
        # Deep copy base config so we don't mutate the shared template
        # F-04: deepcopy is used here; for large configs, consider replacing
        # with a struct-of-immutable-typed-objects pattern to avoid the copy.
        _ws.config = copy.deepcopy(_ws.base_config)
        overrides = _ws.combo_map[combo_idx]

        # Smart probability balancing: if you sweep signal prob, auto-calculate decoy prob
        # NOTE: This runs BEFORE the source is constructed, so we read from
        # the config dict rather than using source.pulse_probabilities().
        # Once the source exists (after build_source() below), downstream
        # code should use source.intensity_config and source.pulse_probabilities()
        # for validated, role-based pulse data.
        _balance_decoy_probabilities(overrides, _ws.config)

        # Apply overrides to config dict
        for key_path, value in overrides.items():
            set_nested_value(_ws.config, key_path, value)

        # Apply detector preset: when detector_type changes via sweep or
        # CLI, override physical parameters with realistic values for
        # that detector type.  Individual sweep entries can still
        # override preset values (applied above via set_nested_value).
        _det_type_preset = str(_ws.config['detector'].get('detector_type', 'SPD')).upper()
        # if _det_type_preset in DETECTOR_PRESETS:
        #     _preset = DETECTOR_PRESETS[_det_type_preset]
        #     # Collect keys that were explicitly swept (they take precedence)
        #     _swept_leaf_keys = set()
        #     for _sp, _ in overrides.items():
        #         _swept_leaf_keys.add(_sp.split('.')[-1])
        #     for _pk, _pv in _preset.items():
        #         if _pk not in _swept_leaf_keys:
        #             _ws.config['detector'][_pk] = _pv
        #             logger.debug(
        #                 'Detector preset %s: %s = %s',
        #                 _det_type_preset, _pk, _pv,
        #             )
        # Rebuild objects for this specific hardware config
        _ws.source = build_source(_ws.config)

        # Validate incompatible combinations
        if (_ws.source.statistics_type == SourceStatisticsType.THERMAL and
            _ws.source.error_model == SourceErrorModel.ADVERSARIAL_BLOCK):
            logger.warning(
                f"Combo {combo_idx}: THERMAL + ADVERSARIAL_BLOCK may produce "
                f"unexpected results. Thermal uses geometric sampling, adversarial "
                f"applies lognormal fluctuation to μ values."
            )

        # F-06: Log warning instead of silently overriding strict_mode.
        # If the user explicitly requested strict_mode=True and the combo
        # is PNRD + dead_time / afterpulse / jitter, the simulation will
        # fail with a ParameterValidationError — which is the correct
        # behaviour.  Previously the code silently set strict_mode=False,
        # hiding the incompatibility from the user.
        #
        # The rewritten detectors module enforces all three of these in
        # strict_mode for PNRD (not just dead_time_ns), so the warning
        # message covers the full set.
        #
        # NOTE: These reads are from the config dict because the detector
        # object hasn't been constructed yet (build_detector() is called
        # below).  Once the detector is built, _validate_params() inside
        # from_config_dict() will catch the same incompatibility and raise
        # ParameterValidationError.  This config-dict check is therefore
        # an *early diagnostic warning* — the authoritative validation
        # happens inside detectors.py._validate_params().
        det_type_str = _ws.config["detector"].get("detector_type", "SPD")
        det_strict = _ws.config["detector"].get("strict_mode", False)
        det_dead = _ws.config["detector"].get("dead_time_ns", 0)
        det_ap = _ws.config["detector"].get("afterpulse_prob", 0)
        det_jitter = _ws.config["detector"].get("jitter_fwhm_ns", 0)
        if (str(det_type_str).upper() == "PNRD" and det_strict
                and (det_dead > 0 or det_ap > 0 or det_jitter > 0)):
            logger.warning(
                "Combo %d: PNRD detector with strict_mode=True and any of "
                "dead_time_ns(%s)/afterpulse_prob(%s)/jitter_fwhm_ns(%s) > 0 "
                "is incompatible with the PNRD path. The simulation may fail "
                "with ParameterValidationError. Set strict_mode=False or zero "
                "the listed fields to avoid this.",
                combo_idx, det_dead, det_ap, det_jitter,
            )

        _ws.detector, _ws.detector_overrides = build_detector(_ws.config)
        _ws.protocol = build_protocol(_ws.config, _ws.source)
        _ws.last_combo_idx = combo_idx
        log_source_details(_ws.source)
        log_detector_details(_ws.detector)

        # F-14: Cache source metadata for this combo
        _ws.source_meta_cache = {combo_idx: get_source_metadata_row(_ws.source)}

    # Use the pre-built worker state
    source = _ws.source
    detector = _ws.detector
    protocol = _ws.protocol
    config = _ws.config
    overrides = _ws.combo_map[combo_idx]

    # --- Auto-optimize pulses per distance (F-29) ---
    # When --auto-optimize-pulses is enabled, run the SLSQP optimizer
    # once per (combo_idx, dist) pair and rebuild the source / protocol
    # with the optimized (mu, nu, p_s, p_d, p_v).  The cache key is
    # (combo_idx, dist) so that the noise=False and noise=True rows at
    # the same (combo, dist) share the same optimized source — this is
    # critical because the optimizer uses the noisy channel model, and
    # we want both rows to use identical pulse parameters for a fair
    # noise-on / noise-off comparison.
    if _ws.auto_optimize_pulses:
        optimize_key = (combo_idx, float(dist))
        if optimize_key != _ws.last_optimized_dist:
            proof_name = config.get("source", {}).get(
                "intended_proof", "LIM_2014"
            )
            try:
                mu_opt, nu_opt, p_s_opt, p_d_opt, p_v_opt = (
                    _optimize_pulses_for_distance(
                        config, float(dist), int(total_pulses), proof_name
                    )
                )
                # Write optimized params back into the config dict.
                # This mutates _ws.config in place; subsequent rows at
                # the same (combo, dist) will reuse the mutated config.
                config["source"]["pulses"]["signal"]["mu"] = mu_opt
                config["source"]["pulses"]["signal"]["prob"] = p_s_opt
                config["source"]["pulses"]["decoy"]["mu"] = nu_opt
                config["source"]["pulses"]["decoy"]["prob"] = p_d_opt
                config["source"]["pulses"]["vacuum"]["mu"] = 0.0
                config["source"]["pulses"]["vacuum"]["prob"] = p_v_opt

                # Rebuild source + protocol with optimized params.
                # Detector is independent of pulse params, so we skip it.
                _ws.source = build_source(config)
                _ws.protocol = build_protocol(config, _ws.source)
                # Refresh cached source metadata so the CSV row records
                # the optimized params, not the combo defaults.
                _ws.source_meta_cache[combo_idx] = (
                    get_source_metadata_row(_ws.source)
                )
                # Update local references
                source = _ws.source
                protocol = _ws.protocol
                _ws.last_optimized_dist = optimize_key
                _ws.optimized_params = (mu_opt, nu_opt, p_s_opt, p_d_opt, p_v_opt)
            except Exception as exc:
                logger.warning(
                    "Auto-optimize failed at combo=%d d=%skm: %s. "
                    "Falling back to combo defaults.",
                    combo_idx, dist, exc,
                )
                _ws.last_optimized_dist = optimize_key
                _ws.optimized_params = None

    # F-05: Derive independent RNG streams from the worker seed using
    # SeedSequence.spawn instead of additive constants (seed+1, seed+2, ...).
    # SeedSequence.spawn guarantees statistical independence between streams,
    # whereas additive constants can produce correlated sequences for some
    # RNG algorithms.
    seed_seq = np.random.SeedSequence(worker_seed)
    child_seqs = seed_seq.spawn(5)
    rng_prepare = np.random.default_rng(child_seqs[0])
    rng_photons = np.random.default_rng(child_seqs[1])
    rng_detect = np.random.default_rng(child_seqs[2])
    rng_sift = np.random.default_rng(child_seqs[3])
    rng_pe = np.random.default_rng(child_seqs[4])

    start_time = time.time()

    try:
        # FIX-PARAMS-ORDER: load_lim2014_dwdm_params called before apply_noise block
        # Load QKDParams for the CURRENT distance BEFORE the apply_noise
        # block.  Previously this was done AFTER detector_run was built,
        # which meant Raman injection read _ws.params from the PREVIOUS
        # iteration (or None on first iteration) and injected Raman for
        # the wrong distance.  See fix_params_loading_order.py.
        # Option C: thread CLI --fiber-loss into the factory so the channel
        # built inside load_lim2014_*_params uses the user-specified
        # fiber_loss_db_km instead of the hardcoded 0.2.  Without this,
        # _compute_raman_dark_rate_hz reads params.channel.fiber_loss_db_km=0.2
        # and the Raman estimate is off by (fiber_loss/0.2) when --fiber-loss
        # is non-default.  No-op when --fiber-loss is left at its default 0.2.
        _fiber_loss_db_km = float(config["channel"].get("fiber_loss_db_km", 0.2))
        if _ws.dwdm:
            try:
                _ws.params = load_lim2014_dwdm_params(distance_km=dist, num_bits=total_pulses,
                                                      fiber_loss_db_km=_fiber_loss_db_km,
                                                      wdm_channel_count=_ws.wdm_channel_count,
                                                      wdm_channel_power_dbm=_ws.wdm_channel_power_dbm)
            except (ValueError, Exception) as _exc_dwdm:
                # distance=0: from_total_loss rejects non-zero loss at zero distance.
                # Fall back to dedicated params, then patch DWDM fields.
                import logging as _log_dwdm
                _log_dwdm.getLogger('main_optimized').debug(
                    f'load_lim2014_dwdm_params failed at distance={dist}: {_exc_dwdm}; falling back'
                )
                _ws.params = load_lim2014_dedicated_params(distance_km=dist, num_bits=total_pulses,
                                                           fiber_loss_db_km=_fiber_loss_db_km)
                # Patch DWDM fields onto the dedicated-params object
                import dataclasses as _dc_dwdm
                _dwdm_patches = {}
                if _ws.wdm_channel_count is not None:
                    _dwdm_patches['wdm_channel_count'] = _ws.wdm_channel_count
                if _ws.wdm_channel_power_dbm is not None:
                    _dwdm_patches['wdm_channel_power_dbm'] = _ws.wdm_channel_power_dbm
                else:
                    # Paper-faithful default: -34 dBm received at distance=0 means
                    # launch = -34 + AWG_loss = -34 + 3 = -31 dBm
                    _awg_loss = float(getattr(_ws.params, 'filter_awg_loss_db', 3.0))
                    _dwdm_patches['wdm_channel_power_dbm'] = -34.0 + _awg_loss
                _dwdm_patches['wdm_raman_coefficient'] = float(
                    _ws.wdm_raman_coefficient if _ws.wdm_raman_coefficient is not None else 1e-9
                )
                _dwdm_patches['filter_awg_loss_db'] = float(
                    getattr(_ws.params, 'filter_awg_loss_db', 3.0)
                )
                _dwdm_patches['filter_model'] = 'awg'
                if hasattr(_ws.params, 'fiber_loss_db_km'):
                    _dwdm_patches['fiber_loss_db_km'] = _fiber_loss_db_km
                if _dwdm_patches:
                    _ws.params = _dc_dwdm.replace(_ws.params, **_dwdm_patches)
        else:
            _ws.params = load_lim2014_dedicated_params(distance_km=dist, num_bits=total_pulses,
                                                       fiber_loss_db_km=_fiber_loss_db_km)

        # FIX-RAMAN-PARAMS-V3: inject Raman into params + stats_map (part 1: params injection)
        # FIX-RAMAN-PARAMS-V3: inject Raman into params + stats_map override: apply CLI coefficient to _ws.params
        if _ws.dwdm and getattr(_ws, "wdm_raman_coefficient", None) is not None:
            try:
                import dataclasses as _dc_raman_ov
                _ov_coef = float(_ws.wdm_raman_coefficient)
                _cur_coef = float(getattr(_ws.params, "wdm_raman_coefficient", 0.0) or 0.0)
                if _cur_coef != _ov_coef:
                    _ws.params = _dc_raman_ov.replace(_ws.params, wdm_raman_coefficient=_ov_coef)
            except Exception as _exc_ov:
                import logging as _log_ov
                _log_ov.getLogger('main_optimized').warning(
                    f'Raman coefficient override failed: {_exc_ov}'
                )

        if _ws.dwdm and apply_noise:
            try:
                _raman_hz_p1 = _compute_raman_dark_rate_hz(_ws.params)
                if _raman_hz_p1 > 0.0:
                    # Inject Raman dark-rate-equivalent (Hz) directly into
                    # params.detector.dark_rate (also Hz; see detectors.py
                    # line 1662 annotation). No per-pulse conversion needed.
                    _det_p1 = getattr(_ws.params, "detector", None)
                    if _det_p1 is not None:
                        _old_dr_p1 = float(getattr(_det_p1, "dark_rate", 0.0) or 0.0)
                        _new_dr_p1 = _old_dr_p1 + _raman_hz_p1
                        import dataclasses as _dc_raman_p1
                        _new_det_p1 = _dc_raman_p1.replace(_det_p1, dark_rate=_new_dr_p1, strict_mode=False)
                        _ws.params = _dc_raman_p1.replace(_ws.params, detector=_new_det_p1)
                        params = _ws.params  # update local var for proof
            except Exception as _exc_raman_p1:
                import logging as _log_raman_p1
                _log_raman_p1.getLogger('main_optimized').warning(
                    f'Raman params injection failed: {_exc_raman_p1}'
                )


        channel = build_channel_from_distance(dist, config)
        # DWDM mode: apply AWG filter loss to the SIMULATION channel too.
        # build_channel_from_distance reads from config["channel"] which does
        # NOT carry filter_model/filter_awg_loss_db.  The params.channel fix
        # (Problem 1) only updates the proof's channel, not the simulation's.
        # Without this, channel_total_loss_db in the CSV is 3 dB too low and
        # detection_yield_signal is too high by a factor of ~2x.
        _total_with_awg = channel.total_loss_db  # default: unchanged
        _awg_patch_needed = False
        if _ws.dwdm:
            _awg_loss_db = getattr(_ws.params, "filter_awg_loss_db", 3.0) if _ws.params else 3.0
            if _awg_loss_db > 0.0:
                _total_with_awg = channel.total_loss_db + _awg_loss_db
                try:
                    channel = FiberChannel.from_total_loss(dist, _total_with_awg)
                except (ValueError, Exception):
                    # distance=0 with non-zero AWG loss: from_total_loss rejects.
                    # Patch ch_params after build_channel_sim_params instead.
                    _awg_patch_needed = True
        # Build frozen ChannelSimParams from validated FiberChannel.
        ch_params = build_channel_sim_params(channel, config)
        # If AWG loss could not be applied to FiberChannel (distance=0),
        # patch ch_params directly (ChannelSimParams is a regular dataclass).
        if _awg_patch_needed:
            import dataclasses as _dc_chp
            ch_params = _dc_chp.replace(
                ch_params,
                total_loss_db=_total_with_awg,
                transmittance=10.0 ** (-_total_with_awg / 10.0),
            )

        # Use source object for pulse_period_ns (validated, frozen state)
        # rather than config dict (mutable, unvalidated).
        # Build SourceSimParams from the validated source object for all
        # source-derived quantities in the simulation loop.
        src_params = build_source_sim_params(source)
        # Defensive check: pulse_period_ns is a @property on OpticalSourceConfig
        # computed from source_rate (1e9 / source_rate).  If source_rate is not
        # correctly set in build_optical_source_config(), pulse_period_ns would
        # be 0.0 or invalid, causing a kernel validation failure in the detector.
        pulse_period_ns = src_params.pulse_period_ns
        if not math.isfinite(pulse_period_ns) or pulse_period_ns <= 0.0:
            raise ConfigurationError(
                f"source.pulse_period_ns = {pulse_period_ns!r} is not strictly positive. "
                f"This indicates that source_rate was not correctly configured — "
                f"pulse_period_ns is derived as 1e9/source_rate via OpticalSourceConfig.pulse_period_ns.",
                context={"pulse_period_ns": pulse_period_ns, "source_rate": src_params.source_rate, "source_type": type(source).__name__},
            )

        prepared_states = protocol.prepare_states(total_pulses, rng_prepare)
        photons = generate_photons_for_prepared_states(source, prepared_states, rng_photons)
        ideal_outcomes_d0 = infer_ideal_outcomes_d0(prepared_states)

        if apply_noise:
            detector_run = detector.clone()
            # Part E: Wire Raman noise into the active detector pipeline.
            # main_optimized.py does NOT use QKDSystem.run_simulation(), so
            # SimulationNoiseMixin._calculate_nonlinear_noise is never invoked.
            # Without this port, wdm_raman_coefficient / wdm_channel_count /
            # wdm_channel_power_dbm are read by NOBODY in the active path.
            # When --dwdm is active, compute the Raman dark-rate-equivalent
            # and bump detector_run.dark_rate (and dark_rate_d1 if present).
            if _ws.dwdm and _ws.params is not None:
                _raman_hz = _compute_raman_dark_rate_hz(_ws.params)
                if _raman_hz > 0.0:
                        # Inject Raman dark-rate-equivalent (Hz) directly into
                        # detector_run.dark_rate / dark_rate_d1 (both Hz; see
                        # detectors.py line 1662 annotation and the threshold
                        # kernel at line 5565 which does the Hz->per-pulse
                        # conversion internally via  wait = -log(u)/(dr0*1e-9)).
                        # No source-rate division needed here.
                    _base_d0 = float(getattr(detector_run, "dark_rate", 0.0) or 0.0)
                    detector_run.dark_rate = _base_d0 + _raman_hz
                    _base_d1 = getattr(detector_run, "dark_rate_d1", None)
                    if _base_d1 is not None:
                        detector_run.dark_rate_d1 = float(_base_d1) + _raman_hz

        else:
            detector_run = detector.noiseless_copy()

        # Use SourceSimParams (src_params) for physics parameters
        # (N_channels, modulation_index, pulse_period_ns) rather than the
        # mutable config dict or direct source object reads.  Channel-derived
        # parameters come from ChannelSimParams (ch_params).  Detector-derived
        # parameters come from DetectorSimParams (det_params).
        # Config-dict reads are kept only for fields that are NOT represented
        # on any SimParams (WDM noise, linewidth, visibility parameters).
        det_kwargs = {
            "N_channels": src_params.N_channels,
            "modulation_index": src_params.modulation_index,
            "mu_signal": source.get_mean_photon_number_by_name("signal"),
            "include_wdm_fwm_noise": config["source"].get("include_wdm_fwm_noise", False),
            "N_channels": src_params.N_channels,
            "distance_km": ch_params.distance_km,
            "dispersion_parameter_ps_nm_km": ch_params.dispersion_parameter_ps_nm_km,
            "linewidth_nm": config["source"].get("linewidth_nm", 0.0),
            "visibility_mismatch_dm": config["source"].get("visibility_mismatch_dm", 0.0),
            "visibility_bias_drift_psi1": config["source"].get("visibility_bias_drift_psi1", 0.0),
            "visibility_bias_drift_psi2": config["source"].get("visibility_bias_drift_psi2", 0.0),
        }
        det_result = detector_run.simulate_detection(
            channel_transmittance=ch_params.transmittance, photon_numbers=photons,
            ideal_outcomes_d0=ideal_outcomes_d0, pulse_period_ns=pulse_period_ns,
            rng=rng_detect, return_diagnostics=True,
            **det_kwargs
        )

        protocol_detection = detector_result_to_protocol_detection_results(det_result, prepared_states.num_pulses)
        detector_meta = protocol_detection.metadata or {}
        sifting_results = protocol.sift_results(prepared_states=prepared_states, detection_results=protocol_detection, rng=rng_sift)
        pe_fraction = float(config["protocol_runtime"]["parameter_estimation_fraction"])
        pe_mask = sample_for_parameter_estimation(sifting_results, rng_pe, pe_fraction)
        summary = sifting_results.summary(confidence_level=0.95)
        stats = tally_counts_from_sifting_results(prepared_states, sifting_results)


        # FIX-RAMAN-PARAMS-V3: inject Raman into params + stats_map (part 2: stats injection)
        # FIX-RAMAN-PARAMS-V3: inject Raman into params + stats_map override: apply CLI coefficient to _ws.params
        if _ws.dwdm and getattr(_ws, "wdm_raman_coefficient", None) is not None:
            try:
                import dataclasses as _dc_raman_ov
                _ov_coef = float(_ws.wdm_raman_coefficient)
                _cur_coef = float(getattr(_ws.params, "wdm_raman_coefficient", 0.0) or 0.0)
                if _cur_coef != _ov_coef:
                    _ws.params = _dc_raman_ov.replace(_ws.params, wdm_raman_coefficient=_ov_coef)
            except Exception as _exc_ov:
                import logging as _log_ov
                _log_ov.getLogger('main_optimized').warning(
                    f'Raman coefficient override failed: {_exc_ov}'
                )

        # Option G: REMOVED Block 3 (analytical stats injection, ~70 lines).
        #
        # This block was duplicating the Raman dark-count injection that
        # Block 2 (lines ~2102-2131 above) already performs via the
        # Monte-Carlo detector sim.  Block 2 bumps detector_run.dark_rate
        # BEFORE detector_run.simulate_detection(), so the sim already
        # produces extra dark clicks that flow into stats.  This block
        # then analytically added MORE Raman-derived counts on top of
        # the already-inflated stats, double-counting the Raman effect.
        #
        # The proof-path Raman injection (Gap [3] fix below, search for
        # "_raman_hz_v4") remains -- it bumps params.detector.dark_rate on
        # the FRESHLY-BUILT late-factory params (the object the proof reads
        # for vacuum yield estimation), which is a separate concern from
        # the Monte-Carlo sim's stats.

        secure_key = 0
        status = "ZERO_KEY"
        # (vacuum-QBER correction no longer needed — detector bug fixed)
        # (signal/decoy QBER correction no longer needed — detector bug fixed)

        proof_name = config["source"].get("intended_proof", "LIM_2014").upper()
        try:
            # DWDM mode: when --dwdm is passed, use the 4-channel WDM preset
            # (Raman noise + AWG filter loss) from qkd.params.load_lim2014_dwdm_params.
            # Otherwise use the single-channel dedicated-fiber preset.
            # Option C: same fiber-loss threading as the early factory call
            # above.  This second factory invocation feeds the proof path,
            # which also reads params.channel.fiber_loss_db_km via
            # _compute_raman_dark_rate_hz.
            _fiber_loss_db_km_v2 = float(config["channel"].get("fiber_loss_db_km", 0.2))
            # _fiber_loss_db_km_v2 instead of _fiber_loss_db_km:
            if _ws.dwdm:
                try:
                    params = load_lim2014_dwdm_params(
                        distance_km=dist, num_bits=total_pulses,
                        wdm_channel_count=_ws.wdm_channel_count,
                        wdm_channel_power_dbm=_ws.wdm_channel_power_dbm,
                        fiber_loss_db_km=_fiber_loss_db_km_v2,
                    )
                except (ValueError, Exception) as _exc_dwdm_v2:
                    import logging as _log_dwdm_v2
                    _log_dwdm_v2.getLogger('main_optimized').debug(
                        f'load_lim2014_dwdm_params (proof) failed at distance={dist}: {_exc_dwdm_v2}'
                    )
                    params = load_lim2014_dedicated_params(distance_km=dist, num_bits=total_pulses,
                                                           fiber_loss_db_km=_fiber_loss_db_km_v2)
                    import dataclasses as _dc_dwdm_v2
                    _dwdm_patches_v2 = {}
                    if _ws.wdm_channel_count is not None:
                        _dwdm_patches_v2['wdm_channel_count'] = _ws.wdm_channel_count
                    if _ws.wdm_channel_power_dbm is not None:
                        _dwdm_patches_v2['wdm_channel_power_dbm'] = _ws.wdm_channel_power_dbm
                    else:
                        _awg_loss_v2 = float(getattr(params, 'filter_awg_loss_db', 3.0))
                        _dwdm_patches_v2['wdm_channel_power_dbm'] = -34.0 + _awg_loss_v2
                    _dwdm_patches_v2['wdm_raman_coefficient'] = float(
                        _ws.wdm_raman_coefficient if _ws.wdm_raman_coefficient is not None else 1e-9
                    )
                    _dwdm_patches_v2['filter_awg_loss_db'] = float(
                        getattr(params, 'filter_awg_loss_db', 3.0)
                    )
                    _dwdm_patches_v2['filter_model'] = 'awg'
                    if hasattr(params, 'fiber_loss_db_km'):
                        _dwdm_patches_v2['fiber_loss_db_km'] = _fiber_loss_db_km_v2
                    if _dwdm_patches_v2:
                        params = _dc_dwdm_v2.replace(params, **_dwdm_patches_v2)
            else:
                params = load_lim2014_dedicated_params(distance_km=dist, num_bits=total_pulses,
                                                       fiber_loss_db_km=_fiber_loss_db_km_v2)
            _ws.params = params  # Part E: cache for Raman injection in detector pipeline
            _ws.params = params  # cache for get_params_metadata_row on error path

            # === Propagate CLI epsilon / f_EC overrides to params ===
            # NOTE: detector params (det_eff, dark_rate, qber_intrinsic,
            # misalignment) are NOT patched here.  Lim2014Proof reads vacuum
            # detections from the stats_map (Monte-Carlo tally counts), not
            # from params.detector; sweeping detector params via the config
            # dict -> build_detector() -> detector_run -> stats_map is the
            # correct and sufficient path.  Raman noise is injected into
            # detector_run.dark_rate directly (lines ~2266-2279), not into
            # params.detector.
            import dataclasses as _dc_v4
            _param_updates = {}

            # CLI-arg override is only available when launched from CLI
            # (main()/argparse).  When launched from the Streamlit UI
            # (qkd_ui.py) there is no argparse namespace, so we skip the
            # override — the values from active_config (DEFAULT_CONFIG
            # patched by the UI) are used as-is.
            try:
                _cli_args = args  # global set by main() in CLI mode
            except NameError:
                _cli_args = None

            if _cli_args is not None:
                for _field_name in ('eps_sec', 'eps_cor', 'eps_pe',
                                    'eps_smooth', 'f_error_correction'):
                    if hasattr(_cli_args, _field_name):
                        _v = getattr(_cli_args, _field_name)
                        if _v is not None:
                            _param_updates[_field_name] = float(_v)

            if _param_updates:
                params = _dc_v4.replace(params, **_param_updates)
                _ws.params = params  # update cache for get_params_metadata_row
            # === end Gap [5] + [6] + [3] fix ===
            # FIX-WDM-CLI: apply --wdm-raman-coefficient override
            if _ws.wdm_raman_coefficient is not None and _ws.dwdm:
                import dataclasses as _dc_fix_wdm
                params = _dc_fix_wdm.replace(params, wdm_raman_coefficient=float(_ws.wdm_raman_coefficient))
                _ws.params = params
            
            
            # اصلاح پارامترها برای اثبات‌های غیر از Lim2014
            if proof_name != "LIM_2014":
                import dataclasses
                
                # ساخت یک دیکشنری از تغییرات مورد نیاز
                updates = {}
                if hasattr(params, 'auto_optimize_lim2014'):
                    updates['auto_optimize_lim2014'] = False
                
                # اضافه کردن پارامترهای امنیتی گمشده (در صورت نیاز)
                if not hasattr(params, 'eps_pa'):
                    updates['eps_pa'] = 1e-10
                if not hasattr(params, 'eps_sif'):
                    updates['eps_sif'] = 1e-10

                # FIX-MA2005-LAYER3: Auto-scale epsilon budget for MA_2005.
                # The default eps_pe=1e-6 >> eps_sec=1e-9 violates the
                # composable security budget (sum must be <= eps_sec).
                # When eps_pe + eps_cor + ... > eps_sec, rescale each
                # component to eps_sec/10 (leaving 40% margin).
                if proof_name == "MA_2005":
                    _eps_sec_val = float(getattr(params, 'eps_sec', 1e-9))
                    _eps_sum = (
                        float(getattr(params, 'eps_pe', 1e-6)) +
                        float(getattr(params, 'eps_cor', 1e-10)) +
                        float(getattr(params, 'eps_smooth', 1e-10)) +
                        float(getattr(params, 'eps_pa', 1e-10))
                        # eps_phase_est is NOT a QKDParams field;
                        # it's derived internally by the proof as eps_pe/10.
                    )
                    if _eps_sum > _eps_sec_val:
                        _eps_each = _eps_sec_val / 10.0
                        updates['eps_pe'] = _eps_each
                        updates['eps_cor'] = _eps_each
                        updates['eps_smooth'] = _eps_each
                        updates['eps_pa'] = _eps_each
                        # Do NOT add eps_phase_est to updates:
                        # QKDParams doesn't have this field.
                        import logging as _log_eps_ma
                        _log_eps_ma.getLogger('main_optimized').info(
                            f'MA2005 epsilon auto-scale: sum({_eps_sum:.2e}) > eps_sec({_eps_sec_val:.2e}), '
                            f'setting each component to {_eps_each:.2e}'
                        )

                # اعمال تغییرات روی شیء frozen با استفاده از replace
                if updates:
                    # Safety: only pass fields that actually exist in QKDParams
                    _valid_fields = {f.name for f in dataclasses.fields(params)}
                    _filtered = {k: v for k, v in updates.items() if k in _valid_fields}
                    _skipped = {k: v for k, v in updates.items() if k not in _valid_fields}
                    if _skipped:
                        logger.warning(
                            f'Skipping non-QKDParams fields in replace(): {_skipped}'
                        )
                    if _filtered:
                        params = dataclasses.replace(params, **_filtered)
            # Auto-scale epsilon budget for WANG_2005 (same issue as MA_2005).
            if proof_name == "WANG_2005":
                _eps_sec_val = float(getattr(params, 'eps_sec', 1e-9))
                _eps_sum_w = (
                    float(getattr(params, 'eps_pe', 1e-6)) +
                    float(getattr(params, 'eps_cor', 1e-10)) +
                    float(getattr(params, 'eps_smooth', 1e-10)) +
                    float(getattr(params, 'eps_pa', 1e-10))
                )
                if _eps_sum_w > _eps_sec_val:
                    _eps_each_w = _eps_sec_val / 10.0
                    _wang_updates = {}
                    for _fn in ('eps_pe', 'eps_cor', 'eps_smooth', 'eps_pa'):
                        if hasattr(params, _fn):
                            _wang_updates[_fn] = _eps_each_w
                    if _wang_updates:
                        import dataclasses as _dc_wang
                        import logging as _log_eps_wang
                        _log_eps_wang.getLogger('main_optimized').info(
                            f'WANG2005 epsilon auto-scale: '
                            f'sum={_eps_sum_w:.2e} > eps_sec={_eps_sec_val:.2e}, '
                            f'each component -> {_eps_each_w:.2e}'
                        )
                        params = _dc_wang.replace(params, **_wang_updates)
                        _ws.params = params
            if proof_name == "MA_2005":
                from qkd.proofs.ma2005 import Ma2005VacuumWeakProof
                # LAYER2 removed: mu/nu/omega/eps_phase_est/n_total are NOT
                # QKDParams fields. The proof reads intensities from
                # params.source.pulse_configs, and eps_phase_est is computed
                # internally by allocate_epsilons() as eps_pe/10.
                proof = Ma2005VacuumWeakProof(params)
            elif proof_name == "WANG_2005":
                from qkd.proofs.wang2005 import Wang2005Proof
                proof = Wang2005Proof(params)
            elif proof_name == "TIGHT":
                from qkd.proofs.tight import BB84TightProof
                proof = BB84TightProof(params)
            else:
                # حالت پیش‌فرض
                proof = Lim2014Proof(params)
                
            # مدیریت تفاوت API در کلاس‌های اثبات امنیتی
            # مدیریت تفاوت API در کلاس‌های اثبات امنیتی
            # NOTE (2024): The phantom vacuum detection bug described below
            # has been FIXED in qkd/detectors.py. The vacuum zero-out workaround
            # is no longer needed and has been REMOVED to prevent regression.
            # (Previously: DISABLED-FIX-VERIFIED vacuum counter zero-out block.)
            if proof_name == "WANG_2005":
                mapped_stats = {
                    "signal": stats.get("0", TallyCounts()),
                    "decoy": stats.get("1", TallyCounts()),
                    "vacuum": stats.get("2", TallyCounts())
                }
                decoy_estimates = proof.estimate_yields_and_errors(stats_map=mapped_stats)
                key_result = proof.calculate_key_length(decoy_estimates, stats_map=mapped_stats)
            elif proof_name == "MA_2005":
                mapped_stats = {
                    "signal": stats.get("0", TallyCounts()),
                    "decoy": stats.get("1", TallyCounts()),
                    "vacuum": stats.get("2", TallyCounts())
                }
                decoy_estimates = proof.estimate_yields_and_errors(stats_map=mapped_stats)
                # Diagnostic: log MA_2005 decoy estimates
                _ma_diag = getattr(decoy_estimates, 'diagnostics', None)
                _ma_nd = getattr(_ma_diag, 'numeric_diagnostics', {}) if _ma_diag else {}
                logger.info(
                    f'MA2005 estimates: Y1_L={decoy_estimates.yield_1_lower_bound:.6e}, '
                    f'e1_U={decoy_estimates.error_rate_1_upper_bound:.6e}, '
                    f'feasible={decoy_estimates.is_feasible}, '
                    f'fail_prob={decoy_estimates.failure_prob_used:.2e}, '
                    f'Y0_L={_ma_nd.get("Y0_L", "N/A")}, '
                    f'Y0_U={_ma_nd.get("Y0_U", "N/A")}, '
                    f'Delta={_ma_nd.get("Delta_tagged_bound_Eq36", "N/A")}'
                )
                key_result = proof.calculate_key_length(decoy_estimates, stats_map=mapped_stats)
            else:
                # Lim2014 و TIGHT (که الان سازگار شد) استخراج را به صورت داخلی انجام می‌دهند
                key_result = proof.calculate_key_length(stats_map=stats)
            secure_key = max(0, int(getattr(key_result, "secure_key_length", 0)))
            status = "OK" if secure_key > 0 else "ZERO_KEY"
        except (LPFailureError, ParameterValidationError, ConfigurationError) as exc:
            qkd_exc = exc
            _log_qkd_exception(qkd_exc, level=logging.WARNING)
            status = f"{qkd_exc.code.value}: {qkd_exc.message[:80]}"
            
            qkd_exc = QKDSimulationError(
                f"{proof_name} secure-key calculation failed for this simulation point.",
                context={"distance_km": dist, "total_pulses": total_pulses, "combo_idx": combo_idx},
                cause=exc,
            )
            _log_qkd_exception(qkd_exc, level=logging.WARNING)
            status = f"{qkd_exc.code.value}: {qkd_exc.message[:80]}"

        sim_time = time.time() - start_time
        total_tally = summarize_tallies(stats)
        stats_0 = stats.get("0", TallyCounts())
        stats_1 = stats.get("1", TallyCounts())
        stats_2 = stats.get("2", TallyCounts())

        # F-14: Use cached source metadata when available
        source_meta = _ws.source_meta_cache.get(combo_idx)
        if source_meta is None:
            source_meta = get_source_metadata_row(source)
            _ws.source_meta_cache[combo_idx] = source_meta

        # F-01: Build detector override row for CSV recording
        det_override_row = _build_detector_override_row(_ws.detector_overrides)

        # Rogers et al. (2007) analytic SBR cross-check (Eq. 15).  Computed
        # from the same (rho_TX, L, tau) used by the Monte-Carlo run so the
        # user can directly compare ``secure_key_rate`` (Monte Carlo) against
        # ``analytic_rogers_sbr`` (paper).  Returns ``None`` when the inputs
        # are out of the analytic model's domain (e.g. dead_time_ns == 0).
        #
        # Uses DetectorSimParams (det_params) and ChannelSimParams (ch_params)
        # for all detector and channel-derived quantities, ensuring parameter
        # consistency.  The link_efficiency helper in channel.py composes
        # eta_ch * eta_det, which is the Rogers "L" parameter.
        det_params = build_detector_sim_params(detector)
        analytic_sbr = _compute_rogers_analytic_sbr(
            pulse_period_ns=pulse_period_ns,
            dead_time_ns=det_params.dead_time_ns,
            channel_transmittance=ch_params.transmittance,
            det_eff_d0=det_params.det_eff_d0,
        )

        # Compute overall QBER, num_errors, and confidence interval from
        # the tally stats.  The detector noise model is now correct
        # (qber_vacuum ≈ 50%, qber_signal ≈ 1%), so no correction is
        # needed.  We keep the protocol's CI width as an approximation
        # of the statistical uncertainty.
        corrected_total_sifted = sum(tc.sifted for tc in stats.values())
        corrected_total_errors = sum(tc.errors_sifted for tc in stats.values())
        if corrected_total_sifted > 0:
            corrected_qber = corrected_total_errors / corrected_total_sifted
        else:
            corrected_qber = 0.0
        # Preserve the CI half-width from the protocol's summary so the
        # interval tracks the corrected point estimate.
        _raw_qber = float(summary.get("qber", 0.0) or 0.0)
        _raw_ci_low = float(summary.get("qber_ci_low", 0.0) or 0.0)
        _raw_ci_high = float(summary.get("qber_ci_high", 0.0) or 0.0)
        _ci_half_width = max(_raw_qber - _raw_ci_low, _raw_ci_high - _raw_qber)
        corrected_ci_low = max(0.0, corrected_qber - _ci_half_width)
        corrected_ci_high = min(1.0, corrected_qber + _ci_half_width)

        # Per-pulse-type corrected stats for the CSV row.
        cstats_0 = stats.get("0", stats_0)
        cstats_1 = stats.get("1", stats_1)

        row = {
            "distance_km": round(dist, 2), "noise_applied": apply_noise, "total_pulses": total_pulses,
            "protocol_name": protocol.protocol_name,
            "protocol_class": config["protocol_runtime"]["protocol_class"],
            "raw_sifted_bits": total_tally.sifted, "secure_key_bits": secure_key,
            "secure_key_rate": round(secure_key / total_pulses, 12) if total_pulses > 0 else 0.0,
            "analytic_rogers_sbr": (
                round(analytic_sbr, 6) if analytic_sbr is not None else None
            ),
            # QBER / num_errors / CI computed from tally stats above.
            # When apply_noise=False, corrected_qber == 0 (noiseless).
            "qber": round(corrected_qber, 8),
            "qber_ci_low": round(corrected_ci_low, 8),
            "qber_ci_high": round(corrected_ci_high, 8),
            "num_errors": int(corrected_total_errors),
            "sifted_z_basis": total_tally.sifted_z, "sifted_x_basis": total_tally.sifted_x,
            "detection_yield_signal": round(stats_0.sifted / stats_0.sent, 10) if stats_0.sent > 0 else 0.0,
            "detection_yield_decoy": round(stats_1.sifted / stats_1.sent, 10) if stats_1.sent > 0 else 0.0,
            "detection_yield_vacuum": round(stats_2.sifted / stats_2.sent, 10) if stats_2.sent > 0 else 0.0,
            # qber_signal/qber_decoy from tally stats (detector model is correct).
            "qber_signal": round(cstats_0.errors_sifted / stats_0.sifted, 8) if stats_0.sifted > 0 else 0.0,
            "qber_decoy": round(cstats_1.errors_sifted / stats_1.sifted, 8) if stats_1.sifted > 0 else 0.0,
            # Report vacuum QBER from tally stats (detector model is correct,
            # qber_vacuum ≈ 50% as expected by Lim2014 Appendix B).
            "qber_vacuum": round(
                stats.get("2", stats_2).errors_sifted
                / stats_2.sifted, 8
            ) if stats_2.sifted > 0 else 0.0,
            # ``diag_double_clicks_discarded`` was the legacy metadata key
            # produced by the pre-rewrite ``DetectionDiagnostics``.  The
            # rewritten diagnostics dataclass no longer has that exact
            # field; ``diag_tossed_events`` now aggregates *all* tossed
            # events (boundary + double-click-discard + kernel-tossed)
            # minus pre-batch and carry-over, exposed via
            # ``DetectionDiagnostics.to_tally_counts().double_clicks_discarded``.
            # We fall back through several keys to stay backward compatible.
            "double_clicks_discarded": int(
                detector_meta.get("diag_double_clicks_discarded")
                or detector_meta.get("diag_tossed_events", 0)
                or total_tally.double_clicks_discarded
            ),
            "parameter_estimation_samples": int(np.count_nonzero(pe_mask)),
            "channel_total_loss_db": round(ch_params.total_loss_db, 8),
            "channel_transmittance": round(ch_params.transmittance, 12),
            "status": status, "simulation_time_sec": round(sim_time, 4),
            **detector_meta,
            **source_meta,
            **det_override_row,  # F-01: detector override values in CSV
            **get_params_metadata_row(params),  # DWDM / filter audit columns
        }

        # DYNAMICALLY ADD SWEPT PARAMETERS TO CSV ROW
        for key_path, value in overrides.items():
            clean_key = "sweep_" + key_path.replace(".", "_")
            row[clean_key] = value

        return row

    except (QKDException, ValueError, TypeError, KeyError, RuntimeError, ArithmeticError) as exc:
        sim_time = time.time() - start_time
        # Preserve the original exception type and message for diagnostics.
        # The previous generic message ("Simulation point failed before a
        # result row could be produced.") hid the root cause in the CSV
        # status field.  Now we include the original exception type and
        # a generous excerpt of its message so that the CSV status column
        # is directly useful for debugging.
        orig_exc_type = type(exc).__name__
        orig_exc_msg = str(exc)
        qkd_exc = _as_qkd_exception(
            exc,
            f"Simulation point failed ({orig_exc_type}: {orig_exc_msg[:200]})",
            context={"distance_km": dist, "total_pulses": total_pulses, "combo_idx": combo_idx, "original_exception_type": orig_exc_type},
        )
        _log_qkd_exception(qkd_exc, level=logging.ERROR)

        # Build source metadata even on error (if source exists).
        source_meta: Dict[str, Any] = {}
        if _ws.source is not None:
            try:
                source_meta = get_source_metadata_row(_ws.source)
            except (QKDException, ValueError, TypeError, KeyError, RuntimeError) as meta_exc:
                logger.debug("Could not build source metadata for failed row.", exc_info=meta_exc)
                source_meta = {"source_type": "unknown"}

        # F-07: Error-path CSV row now includes protocol_class, detector metadata
        # columns, and detector override columns so that every row has the same
        # field set regardless of success or failure.
        detector_meta: Dict[str, Any] = {}
        det_override_row: Dict[str, Any] = {}
        if _ws.detector_overrides:
            det_override_row = _build_detector_override_row(_ws.detector_overrides)

        protocol_class = config.get("protocol_runtime", {}).get("protocol_class", "unknown") if config else "unknown"

        row = {
            "distance_km": round(dist, 2), "noise_applied": apply_noise, "total_pulses": total_pulses,
            "protocol_name": getattr(_ws.protocol, "protocol_name", "unknown") if _ws.protocol else "unknown",
            "protocol_class": protocol_class,
            "raw_sifted_bits": 0, "secure_key_bits": 0, "secure_key_rate": 0.0,
            "analytic_rogers_sbr": None,  # not computed on the error path
            "qber": 0.0, "qber_ci_low": 0.0, "qber_ci_high": 0.0, "num_errors": 0,
            "sifted_z_basis": 0, "sifted_x_basis": 0,
            "detection_yield_signal": 0.0, "detection_yield_decoy": 0.0, "detection_yield_vacuum": 0.0,
            "qber_signal": 0.0, "qber_decoy": 0.0, "qber_vacuum": 0.0,
            "double_clicks_discarded": 0, "parameter_estimation_samples": 0,
            "channel_total_loss_db": None, "channel_transmittance": None,
            "status": f"{qkd_exc.code.value}: {qkd_exc.message[:120]}", "simulation_time_sec": round(sim_time, 4),
            **detector_meta,
            **source_meta,
            **det_override_row,  # F-07/F-01: override columns on error rows too
            **get_params_metadata_row(_ws.params),  # DWDM columns on error rows too
        }
        for key_path, value in overrides.items():
            row["sweep_" + key_path.replace(".", "_")] = value
        return row


# ============== MAIN FUNCTION ==============
def run_and_save_csv(config: Dict[str, Any], num_workers: int = None, sweeps: Dict = None, auto_optimize_pulses: bool = False, dwdm: bool = False, wdm_channel_count: int = 4, wdm_channel_power_dbm = None, wdm_raman_coefficient: float = None):
    # Configure logging so that ERROR-level messages from failed simulation
    # points are visible in the terminal.  Without this, the default root
    # logger level is WARNING and the detailed exception information logged
    # by _log_qkd_exception() is silently discarded.
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    print("--- Running QKD Simulations (SWEEP MODE) ---")

    cpu_count = mp.cpu_count() or 1
    default_workers = 8

    if num_workers is None:
        num_workers = default_workers
    if not isinstance(num_workers, int) or num_workers < 1:
        raise ParameterValidationError(
            "num_workers must be a positive integer.",
            param_name="num_workers",
            param_value=num_workers,
        )

    if num_workers > cpu_count:
        print(f"[WARNING] Requested {num_workers} workers but only {cpu_count} CPUs available.")

    # F-27: Guard psutil import with try/except so the script still runs
    # when psutil is not installed (it was a conditional import before).
    MEMORY_PER_WORKER_MB = 500
    if num_workers > 16:
        try:
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024**3)
            required_ram_gb = (num_workers * MEMORY_PER_WORKER_MB) / 1024
            if required_ram_gb > available_ram_gb * 0.8:
                print(f"[WARNING] {num_workers} workers may need ~{required_ram_gb:.1f}GB RAM. Risk of OOM!")
        except ImportError:
            logger.warning(
                "psutil not installed; cannot check available RAM for %d workers.",
                num_workers,
            )

    print(f"[INFO] CPU Count: {cpu_count}")
    print(f"[INFO] Using {num_workers} worker processes")

    sim_cfg = config["simulation"]
    if int(sim_cfg["min_pulses_log"]) > int(sim_cfg["max_pulses_log"]):
        raise ParameterValidationError(
            "min_pulses_log must be less than or equal to max_pulses_log.",
            param_name="simulation.min_pulses_log",
            param_value=sim_cfg["min_pulses_log"],
            context={"max_pulses_log": sim_cfg["max_pulses_log"]},
        )
    if int(sim_cfg["distance_points"]) < 1:
        raise ParameterValidationError(
            "distance_points must be at least 1.",
            param_name="simulation.distance_points",
            param_value=sim_cfg["distance_points"],
        )

    seed = sim_cfg["rng_seed"]
    OpticalSource.validate_rng_seed(seed)

    max_pulses = 10 ** config["simulation"]["max_pulses_log"]
    min_block_for_low_yield = OpticalSource.calculate_minimum_block_size(expected_yield=0.001)
    if max_pulses < min_block_for_low_yield:
        logger.warning(f"max_pulses ({max_pulses}) below minimum block size ({min_block_for_low_yield}).")

    source_tmp = build_source(config)
    protocol_name = build_protocol(config, source_tmp).protocol_name

    sweeps = sweeps if sweeps else {}
    if not isinstance(sweeps, dict):
        raise ConfigurationError(
            "sweeps must be a dictionary mapping dotted config keys to lists of values.",
            context={"value_type": type(sweeps).__name__},
        )
    for sweep_key, values in sweeps.items():
        if not isinstance(sweep_key, str) or not sweep_key:
            raise ConfigurationError(
                "Each sweep key must be a non-empty dotted string.",
                context={"sweep_key": sweep_key},
            )
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ConfigurationError(
                "Each sweep value must be a non-empty list or tuple.",
                context={"sweep_key": sweep_key, "value_type": type(values).__name__},
            )

    sweep_keys = list(sweeps.keys())
    sweep_values = list(sweeps.values())
    combinations = list(itertools.product(*sweep_values))

    # F-15: Warn when the Cartesian product produces an excessive number of
    # combinations (default threshold 10 000).  This guards against accidental
    # combinatorial explosion from adding new sweep keys.
    MAX_COMBOS_WARNING = 10_000
    if len(combinations) > MAX_COMBOS_WARNING:
        print(
            f"[WARNING] Sweep produces {len(combinations):,} hardware combinations "
            f"(threshold: {MAX_COMBOS_WARNING:,}). Consider reducing sweep keys."
        )

    combo_map = {i: dict(zip(sweep_keys, combo)) for i, combo in enumerate(combinations)}
    print(f"[INFO] Generated {len(combo_map)} hardware configuration combinations.")

    distances = np.linspace(config["simulation"]["distance_start_km"], config["simulation"]["distance_stop_km"], config["simulation"]["distance_points"])
    pulse_counts = [10 ** i for i in range(config["simulation"]["min_pulses_log"], config["simulation"]["max_pulses_log"] + 1)]

    work_items = []
    base_rng = np.random.default_rng(seed)
    for combo_idx in combo_map.keys():
        for apply_noise in [False, True]:
            for total_pulses in pulse_counts:
                for dist in distances:
                    worker_seed = int(base_rng.integers(0, 2**63))
                    work_items.append((dist, total_pulses, apply_noise, worker_seed, combo_idx))

    print(f"[INFO] Total simulations to run: {len(work_items):,}")
    if auto_optimize_pulses:
        print(f"[INFO] Auto-optimize pulses: ENABLED (per-distance SLSQP optimization)")

    sample_sweep_keys = ["sweep_" + k.replace(".", "_") for k in sweeps.keys()]
    source_meta_keys = list(get_source_metadata_row(source_tmp).keys())

    fieldnames = [
        "distance_km", "noise_applied", "total_pulses", "protocol_name", "protocol_class",
        "raw_sifted_bits", "secure_key_bits", "secure_key_rate",
        # Rogers et al. (2007) analytic cross-check (Eq. 15).  Populated
        # for every successful simulation point; ``None`` on the error path
        # and when the inputs are out of the analytic model's domain.
        "analytic_rogers_sbr",
        "qber", "qber_ci_low", "qber_ci_high", "num_errors",
        "sifted_z_basis", "sifted_x_basis",
        "detection_yield_signal", "detection_yield_decoy", "detection_yield_vacuum",
        "qber_signal", "qber_decoy", "qber_vacuum",
        "double_clicks_discarded", "parameter_estimation_samples",
        "channel_total_loss_db", "channel_transmittance",
        "status", "simulation_time_sec",
    ] + DETECTOR_METADATA_KEYS + source_meta_keys + DETECTOR_OVERRIDE_KEYS + PARAMS_METADATA_KEYS + sample_sweep_keys
    fieldnames = list(dict.fromkeys(fieldnames))

    csv_filename = "qkd_results_optimized_sweep.csv"
    progress_interval = max(1, len(work_items) // 20) if work_items else 1
    chunksize = max(1, min(100, len(work_items) // max(1, num_workers * 8))) if work_items else 1

    print(f"[INFO] Protocol selected: {protocol_name}")
    print(f"[INFO] Writing results to '{csv_filename}'")

    completed = 0

    def _on_row(row):
        # Closure over `writer`, `f`, `progress_interval`, `work_items` —
        # behavior is identical to the previous inline loop body.
        nonlocal completed
        writer.writerow(row)
        completed += 1
        if (completed == 1
                or completed == len(work_items)
                or completed % progress_interval == 0):
            print(f"[INFO] Completed {completed:,}/{len(work_items):,} simulations")
            f.flush()

    with open_atomic_text(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        # Dispatch is delegated to main_optimized_dispatch.dispatch_work_items.
        # The single-worker path calls init_worker + a sequential for-loop
        # (order preserved); the multi-worker path uses mp.Pool.imap_unordered
        # (order NOT preserved, by design) with the `with` block acting as a
        # completion barrier.  See tests/test_main_optimized_concurrency.py.
        dispatch_work_items(
            work_items=work_items,
            worker_fn=run_single_simulation,
            num_workers=num_workers,
            init_fn=init_worker,
            init_args=(config, combo_map, auto_optimize_pulses, dwdm, wdm_channel_count, wdm_channel_power_dbm, wdm_raman_coefficient),
            on_row=_on_row,
            chunksize=chunksize,
        )

    print(f"\n[SUCCESS] Results saved to '{csv_filename}'.")


if __name__ == "__main__":
    import traceback

    try:
        parser = argparse.ArgumentParser(description="Run QKD simulations (SWEEP MODE)")

        # --- Existing Arguments ---
        parser.add_argument("--protocol-class", type=str, default="BB84DecoyProtocol",
                            choices=["BB84DecoyProtocol", "B92Protocol", "MDIQKDProtocol", "RedundantTransmissionProtocol"])
        parser.add_argument("--fiber-loss", type=float, default=0.2)
        parser.add_argument("--min-pulses-log", type=int, default=4)
        parser.add_argument("--max-pulses-log", type=int, default=6)
        parser.add_argument("--rng-seed", type=int, default=12345)
        parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count - 1)")
        parser.add_argument("--source-class", type=str, default="optical")
        parser.add_argument("--alice-z-basis-prob", type=float, default=0.5)
        parser.add_argument("--bob-z-basis-prob", type=float, default=0.5)
        parser.add_argument("--z-basis-prob", type=float, default=0.5)
        parser.add_argument("--protocol-double-click-policy", type=str, default="DISCARD")
        parser.add_argument("--parameter-estimation-fraction", type=float, default=0.1)
        parser.add_argument("--det-eff-d0", type=float, default=0.15)
        parser.add_argument("--det-eff-d1", type=float, default=0.15)
        parser.add_argument("--dark-rate", type=float, default=600.0)  # paper-faithful (Lim2014)
        parser.add_argument("--qber-intrinsic", type=float, default=0.005)  # paper-faithful (Lim2014)
        parser.add_argument("--misalignment", type=float, default=0.0)
        parser.add_argument("--distance-start-km", type=float, default=0.0)
        parser.add_argument("--distance-stop-km", type=float, default=150.0)
        parser.add_argument("--distance-points", type=int, default=16)
        parser.add_argument("--detector-type", type=str, default="SPD",
                            choices=["SPD", "SNSPD", "PNRD"])
        # Rogers et al. (2007) integration: detector-level double-click
        # policy.  This is distinct from the protocol-level policy above
        # (which controls how the protocol handles simultaneous D0&D1
        # clicks at sifting time).  The detector-level policy controls
        # how the *detector itself* resolves double-clicks and, when
        # 'rogers_2007' is selected, applies the Rogers-style
        # sequence-collapsing sifting rule (at most one sifted bit per
        # dead-time-spanning detection sequence; requires dead_time_ns > 0).
        parser.add_argument(
            "--detector-double-click-policy",
            type=str,
            default=None,
            choices=["RANDOM", "DISCARD", "rogers_2007"],
            help=(
                "Override the detector-level double-click policy.  'RANDOM' "
                "keeps one click at random; 'DISCARD' drops both.  "
                "'rogers_2007' enables the Rogers et al. (2007) "
                "sequence-collapsing sifting rule: at most one sifted bit "
                "per dead-time-spanning detection sequence (requires "
                "dead_time_ns > 0 to have any effect).  Default: leave "
                "unchanged (uses the value in DEFAULT_CONFIG)."
            ),
        )
        parser.add_argument(
            "--dead-time-ns",
            type=float,
            default=None,
            help="Override detector dead time in ns (default 10.0).  Required "
                 "to be > 0 for the 'rogers_2007' sifting policy to have "
                 "any effect.",
        )

        # --- Sweep Argument ---
        parser.add_argument("--sweep-json", type=str, default=None,
                            help='JSON string of parameters to sweep. Example: \'{"source.pulses.signal.mu": [0.3, 0.5], "detector.det_eff_d0": [0.1, 0.2]}\'')
        # --- Additional Source Arguments ---
        parser.add_argument("--statistics-type", type=str, default=None,
                            choices=["POISSON", "THERMAL"],
                            help="Override source statistics type")
        parser.add_argument("--error-model", type=str, default=None,
                            choices=["RANDOM_GAUSSIAN", "ADVERSARIAL_BLOCK"],
                            help="Override source error model")
        parser.add_argument("--intended-proof", type=str, default=None,
                            choices=["LIM_2014", "MA_2005", "WANG_2005", "TIGHT"],
                            help="Override security proof (default: LIM_2014 from DEFAULT_CONFIG)")
        parser.add_argument("--confidence-method", type=str, default=None,
                            choices=["CLOPPER_PEARSON", "HOEFFDING", "GAUSSIAN"],   # must match ConfidenceBoundMethod members
                            help="Override confidence bound method (default: GAUSSIAN from DEFAULT_CONFIG)")
        parser.add_argument("--ideal-emission-probability", type=float, default=None,
                            help="Override ideal emission probability (0.0-1.0)")
        parser.add_argument("--n-channels", type=int, default=None,
                            help="Number of channels (default from config)")
        parser.add_argument("--use-linear-modulation", action="store_true", default=None,
                            help="Use linear modulation approximation")
        parser.add_argument("--is-bidirectional", action="store_true", default=None,
                            help="Enable bidirectional mode")
        parser.add_argument("--auto-optimize-pulses", action="store_true", default=False,
                            help="Run per-distance pulse-parameter optimizer (mu, nu, p_s, p_d, p_v) "
                                 "using asymptotic GLLP rate with finite-size penalty. "
                                 "Each (combo, distance) gets its own optimized params.")
        parser.add_argument("--dwdm", action="store_true", default=False,
                            help="Use the Lim2014 DWDM preset (4-channel WDM + Raman noise + "
                                 "AWG filter loss) instead of the single-channel dedicated-fiber "
                                 "preset. Loads params via qkd.params.load_lim2014_dwdm_params.")

        # --- Security parameter overrides (Gap [5] fix) ---
        parser.add_argument("--eps-sec", type=float, default=None,
                            help="Override eps_sec (default: 1e-7 from factory)")
        parser.add_argument("--eps-cor", type=float, default=None,
                            help="Override eps_cor (default: 1e-7 from factory)")
        parser.add_argument("--eps-pe", type=float, default=None,
                            help="Override eps_pe (default: 1e-7 from factory)")
        parser.add_argument("--eps-smooth", type=float, default=None,
                            help="Override eps_smooth (default: 1e-7 from factory)")
        parser.add_argument("--f-error-correction", type=float, default=None,
                            help="Override f_error_correction (default: 1.16 from factory)")
        parser.add_argument("--wdm-channel-count", type=int, default=4, metavar="N",
                            help="Number of classical WDM channels (default: 4, paper's 4+1 arch). "
                                 "Only used when --dwdm is active. The quantum channel is separate "
                                 "and not counted. Try 8, 16, 32 to study Raman scaling.")
        parser.add_argument("--wdm-channel-power-dbm", type=float, default=None, metavar="P",
                            help="Per-channel LAUNCH power in dBm (default: None = paper-faithful "
                                 "distance-aware, holding RECEIVED power at -34 dBm per Lim2014 "
                                 "Section IV and Ref. [38]). Only used when --dwdm is active. "
                                 "Pass a float (e.g., -3.0) to force a constant launch power "
                                 "across all distances (legacy sweep mode).")
        # FIX-RAMAN-COEF: --wdm-raman-coefficient CLI flag wired
        parser.add_argument("--wdm-raman-coefficient", type=float, default=None,
                            help="Raman scattering coefficient (default 1e-9 from params.py). ")
        args = parser.parse_args()

        # Build the base configuration from arguments
        active_config = {
            "protocol_params": dict(DEFAULT_CONFIG["protocol_params"]),
            "protocol_runtime": dict(DEFAULT_CONFIG["protocol_runtime"]),
            "channel": dict(DEFAULT_CONFIG["channel"]),
            "detector": dict(DEFAULT_CONFIG["detector"]),
            "source": dict(DEFAULT_CONFIG["source"]),
            "protocol": {"epsilons": dict(DEFAULT_CONFIG["protocol"]["epsilons"])},
            "simulation": dict(DEFAULT_CONFIG["simulation"]),
        }

        active_config["protocol_runtime"]["protocol_class"] = args.protocol_class
        # Apply source overrides from command line
        if args.statistics_type is not None:
            active_config["source"]["statistics_type"] = args.statistics_type
        if args.error_model is not None:
            active_config["source"]["error_model"] = args.error_model
        if args.intended_proof is not None:
            active_config["source"]["intended_proof"] = args.intended_proof
        if args.confidence_method is not None:
            active_config["source"]["confidence_method"] = args.confidence_method
        if args.ideal_emission_probability is not None:
            active_config["source"]["ideal_emission_probability"] = args.ideal_emission_probability
        if args.n_channels is not None:
            active_config["source"]["N_channels"] = args.n_channels
        if args.use_linear_modulation is not None:
            active_config["source"]["use_linear_modulation_approximation"] = args.use_linear_modulation
        if args.is_bidirectional is not None:
            active_config["source"]["is_bidirectional"] = args.is_bidirectional
        active_config["protocol_runtime"]["alice_z_basis_prob"] = args.alice_z_basis_prob
        active_config["protocol_runtime"]["bob_z_basis_prob"] = args.bob_z_basis_prob
        active_config["protocol_runtime"]["z_basis_prob"] = args.z_basis_prob
        active_config["protocol_runtime"]["double_click_policy"] = args.protocol_double_click_policy
        active_config["protocol_runtime"]["parameter_estimation_fraction"] = args.parameter_estimation_fraction
        active_config["channel"]["fiber_loss_db_km"] = args.fiber_loss
        active_config["simulation"]["min_pulses_log"] = args.min_pulses_log
        active_config["simulation"]["max_pulses_log"] = args.max_pulses_log
        active_config["simulation"]["rng_seed"] = args.rng_seed
        active_config["simulation"]["distance_start_km"] = args.distance_start_km
        active_config["simulation"]["distance_stop_km"] = args.distance_stop_km
        active_config["simulation"]["distance_points"] = args.distance_points
        active_config["detector"]["det_eff_d0"] = args.det_eff_d0
        active_config["detector"]["det_eff_d1"] = args.det_eff_d1
        active_config["detector"]["dark_rate"] = args.dark_rate
        active_config["detector"]["qber_intrinsic"] = args.qber_intrinsic
        active_config["detector"]["misalignment"] = args.misalignment
        active_config["detector"]["detector_type"] = args.detector_type
        active_config["protocol_params"]["detector_type"] = args.detector_type
        # Rogers et al. (2007) integration: apply detector-level policy
        # and dead-time overrides from the CLI.  These are the two knobs
        # that select the new sequence-collapsing sifting rule.
        if args.detector_double_click_policy is not None:
            active_config["detector"]["double_click_policy"] = (
                args.detector_double_click_policy
            )
        if args.dead_time_ns is not None:
            active_config["detector"]["dead_time_ns"] = args.dead_time_ns

        # Determine which sweeps to run:
        if args.sweep_json:
            current_sweeps = parse_json_strict(
                args.sweep_json,
                expected_type=dict,
                error_message=(
                    "--sweep-json must be valid JSON. Provide an object like "
                    '{"source.pulses.signal.mu": [0.3, 0.5]}.'
                ),
            )
            print(f"[INFO] Loaded custom sweeps from CLI.")
        else:
            current_sweeps = DEFAULT_SWEEPS
            print(f"[INFO] Using DEFAULT_SWEEPS defined in script.")

        # Pass both the config and the sweeps dictionary to the runner
        run_and_save_csv(active_config, num_workers=args.workers, sweeps=current_sweeps,
                         auto_optimize_pulses=args.auto_optimize_pulses,
                         dwdm=args.dwdm,
                         wdm_channel_count=args.wdm_channel_count,
                         wdm_channel_power_dbm=args.wdm_channel_power_dbm,

                         wdm_raman_coefficient=args.wdm_raman_coefficient)
    except KeyboardInterrupt as exc:
        interrupted = SimulationInterruptedError(cause=exc)
        _log_qkd_exception(interrupted, level=logging.WARNING)
        print(f"\n{interrupted}")
        raise SystemExit(130) from exc
    except QKDException as exc:
        _log_qkd_exception(exc, level=logging.ERROR)
        print("\n" + "="*60)
        print("QKD ERROR OCCURRED:")
        print("="*60)
        print(exc)
        if exc.context:
            print(safe_json_dumps(exc.context))
        print("="*60)
        raise SystemExit(1) from exc
    except (ValueError, TypeError, KeyError, RuntimeError, ArithmeticError) as exc:
        wrapped = _as_qkd_exception(
            exc,
            "Unexpected failure while running main_optimized.py.",
            context={"entrypoint": "__main__"},
        )
        _log_qkd_exception(wrapped, level=logging.ERROR)
        print("\n" + "="*60)
        print("FATAL ERROR OCCURRED:")
        print("="*60)
        traceback.print_exception(type(wrapped), wrapped, wrapped.__traceback__)
        print("="*60)
        raise SystemExit(1) from exc
