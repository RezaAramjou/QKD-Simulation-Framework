"""
qkd_ui.py — Production-grade Streamlit frontend for the QKD simulation engine
============================================================================

A complete refactor of the original qkd_ui.py providing:

  • Full parameter parity with ``SIMULATION_SPEC.md`` — every knob in
    ``DEFAULT_CONFIG`` plus CLI-only flags such as ``--dwdm`` and
    ``--auto-optimize-pulses`` is controllable from the UI, with realistic
    sliders, scientific-notation number inputs, and live validation warnings.
  • Three execution modes — **Single Point**, **1-D Sweep**, and **2-D Sweep**
    — sharing one configuration surface, with KPI summary cards for the
    single-point run and a real progress bar for sweep runs.
  • A clean bridge to the core engine: UI inputs are packaged into the
    exact dict + sweeps dict expected by ``main_optimized.run_and_save_csv()``
    and the DWDM/auto-optimize flags are forwarded as keyword arguments.
    Single-point runs are wrapped in ``@st.cache_data`` for responsiveness.
  • Academic-styled Plotly visualisations covering every plot in spec §4.2:
    SKR-vs-distance, QBER-vs-distance (with CI band), detection yield,
    channel loss budget (DWDM twin axis), finite-size 2-D contour, sweep
    sensitivity (with argmax marker), and a convergence audit.  Every axis
    is labelled with physical units and LaTeX-style ticks; SKR/QBER/yield
    default to log10.
  • Export section — download the filtered sweep table as CSV, and any
    plot as PNG (300 dpi) or PDF (vector) via kaleido.

Run with::

    streamlit run qkd_ui.py

The UI degrades gracefully if the engine package (``main_optimized``,
``qkd.*``) is unavailable — a banner is shown and parameter editing still
works in dry-run mode for tutorial / inspection purposes.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import concurrent.futures
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────
# Engine imports — wrapped so the UI boots even when the engine is absent
# ─────────────────────────────────────────────────────────────────────────
try:
    from main_optimized import DEFAULT_CONFIG, run_and_save_csv
    from qkd.proofs.optimization import (
        optimize_decoy_intensities,
        calculate_optimal_mu_ma2005,
        optimize_rotation_angle,
        optimize_rotation_angle_chapman,
    )
    try:
        from qkd.channel import MAX_DISTANCE_KM, MAX_FIBER_LOSS_DB_KM
    except Exception:
        MAX_DISTANCE_KM = 500.0
        MAX_FIBER_LOSS_DB_KM = 5.0
    _ENGINE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover — UI keeps running in dry-run mode
    DEFAULT_CONFIG: dict[str, Any] = {}  # type: ignore[no-redef]
    run_and_save_csv = None  # type: ignore[assignment]
    optimize_decoy_intensities = None  # type: ignore[assignment]
    calculate_optimal_mu_ma2005 = None  # type: ignore[assignment]
    optimize_rotation_angle = None  # type: ignore[assignment]
    optimize_rotation_angle_chapman = None  # type: ignore[assignment]
    MAX_DISTANCE_KM = 500.0
    MAX_FIBER_LOSS_DB_KM = 5.0
    _ENGINE_IMPORT_ERROR = exc


# ─────────────────────────────────────────────────────────────────────────
# Application metadata
# ─────────────────────────────────────────────────────────────────────────
APP_TITLE = "QKD Simulation Dashboard"
APP_SUBTITLE = (
    "Production frontend for ``main_optimized.run_and_save_csv()`` — "
    "complete parameter parity with SIMULATION_SPEC.md"
)
DEFAULT_OUTPUT_DIR = "qkd_ui_runs"
ENGINE_CSV_NAME = "qkd_results_optimized_sweep.csv"

# NOTE: The finite-size axis (min_pulses_log → max_pulses_log) is enabled
# natively by setting max_pulses_log > min_pulses_log on the Simulation tab;
# it is NOT exposed as a synthetic sweep key (the engine rejects unknown keys).


# ─────────────────────────────────────────────────────────────────────────
# Sweep catalog (master list of pre-canned sweep axes)
# ─────────────────────────────────────────────────────────────────────────
MASTER_SWEEP_CATALOG: dict[str, dict[str, Any]] = {
    # ── Source pulses ──────────────────────────────────────────────────────
    "source.pulses.signal.mu":   {"default": [0.3, 0.5, 0.8],                          "group": "Source"},
    "source.pulses.decoy.mu":    {"default": [0.05, 0.1, 0.2],                         "group": "Source"},
    # ── Detector efficiency ────────────────────────────────────────────────
    "detector.det_eff_d0":       {"default": [0.10, 0.15, 0.20],                        "group": "Detector"},
    "detector.det_eff_d1":       {"default": [0.10, 0.15, 0.20],                        "group": "Detector"},
    # ── Detector noise ──────────────────────────────────────────────────────
    "detector.dark_rate":        {"default": [1e-7, 1e-6, 1e-5],                        "group": "Detector"},
    "detector.qber_intrinsic":   {"default": [0.005, 0.01, 0.02],                      "group": "Detector"},
    "detector.misalignment":     {"default": [0.002, 0.005, 0.01],                     "group": "Detector"},
    # ── Detector advanced ──────────────────────────────────────────────────
    "detector.bias_voltage":            {"default": [44.0, 50.0],                          "group": "Detector"},
    "detector.temperature_k":           {"default": [77.0, 293.0],                        "group": "Detector"},
    "detector.afterpulse_lifetime_ns":  {"default": [10.0, 100.0, 500.0],                 "group": "Detector"},
    "detector.detector_type":           {"default": ["SPD", "PNRD", "SNSPD"],             "group": "Detector"},
    "detector.dead_time_model":         {"default": ["NON_PARALYZABLE"],                  "group": "Detector"},
    "detector.afterpulse_model":        {"default": ["EXPONENTIAL", "GEOMETRIC"],         "group": "Detector"},
    "detector.double_click_policy":      {"default": ["RANDOM", "DISCARD", "rogers_2007"], "group": "Detector"},
    "detector.strict_mode":              {"default": [False, True],                       "group": "Detector"},
    # ── Source properties ──────────────────────────────────────────────────
    "source.intensity_jitter":          {"default": [0.0, 0.05, 0.1, 0.15],              "group": "Source"},
    "source.modulation_index":          {"default": [0.90, 0.95, 0.99, 1.0],            "group": "Source"},
    "source.statistics_type":           {"default": ["POISSON", "THERMAL"],              "group": "Source"},
    "source.error_model":               {"default": ["RANDOM_GAUSSIAN", "ADVERSARIAL_BLOCK"], "group": "Source"},
    "source.intended_proof":            {"default": ["LIM_2014", "MA_2005", "WANG_2005", "TIGHT"], "group": "Source"},
    "source.confidence_method":         {"default": ["GAUSSIAN", "HOEFFDING", "CLOPPER_PEARSON"], "group": "Source"},
    "source.ideal_emission_probability":{"default": [0.95, 0.99, 1.0],                  "group": "Source"},
    "source.use_linear_modulation_approximation": {"default": [False, True],            "group": "Source"},
    "source.N_channels":                {"default": [1, 2, 4],                           "group": "Source"},
    "source.is_bidirectional":          {"default": [False, True],                       "group": "Source"},
    # ── Channel ─────────────────────────────────────────────────────────────
    "channel.dispersion_parameter_ps_nm_km": {"default": [0.0, 17.0],                    "group": "Channel"},
    "channel.fiber_loss_db_km":              {"default": [0.18, 0.20, 0.22],              "group": "Channel"},
    # ── Protocol ────────────────────────────────────────────────────────────
    "protocol_runtime.protocol_class":     {"default": ["BB84DecoyProtocol", "RedundantTransmissionProtocol"], "group": "Protocol"},
    "protocol_runtime.use_entangling_decoder": {"default": [False, True],                "group": "Protocol"},
    "protocol_runtime.redundancy_M":       {"default": [2, 3],                           "group": "Protocol"},
    "protocol_params.error_correction_efficiency": {"default": [1.10, 1.16, 1.20],       "group": "Protocol"},
    "protocol_runtime.parameter_estimation_fraction": {"default": [0.05, 0.10, 0.20],    "group": "Protocol"},
    # ── Epsilons ────────────────────────────────────────────────────────────
    "protocol.epsilons.eps_sec":           {"default": [1e-10, 1e-9, 1e-8],             "group": "Epsilon"},
    "protocol.epsilons.eps_cor":           {"default": [1e-11, 1e-10, 1e-9],            "group": "Epsilon"},
}

DEFAULT_SWEEP: dict[str, list[Any]] = {
    "source.pulses.signal.mu": [0.3, 0.5, 0.8],
    "detector.det_eff_d0":     [0.10, 0.15, 0.20],
    "detector.dark_rate":      [1e-7, 1e-6, 1e-5],
}

# 1-D sweep candidates recommended by spec §2 (good single-axis parameters)
SWEEP_1D_CANDIDATES: list[str] = [
    "source.pulses.signal.mu",
    "source.pulses.decoy.mu",
    "detector.det_eff_d0",
    "detector.dark_rate",
    "detector.qber_intrinsic",
    "detector.misalignment",
    "source.intensity_jitter",
    "source.modulation_index",
    "protocol_params.error_correction_efficiency",
    "protocol_runtime.alice_z_basis_prob",
    "protocol_runtime.parameter_estimation_fraction",
    "channel.fiber_loss_db_km",
]

# 2-D sweep candidates — paired with distance by default (spec §2.1.2, §2.2.2)
SWEEP_2D_CANDIDATES: list[str] = [
    "source.pulses.signal.mu",
    "source.pulses.decoy.mu",
    "detector.det_eff_d0",
    "detector.dark_rate",
    "detector.qber_intrinsic",
    "protocol_params.error_correction_efficiency",
]
# NOTE: The finite-size axis (min_pulses_log → max_pulses_log) is enabled
# natively by setting max_pulses_log > min_pulses_log on the Simulation tab;
# it is NOT exposed as a synthetic sweep key (the engine rejects unknown keys).


# ─────────────────────────────────────────────────────────────────────────
# Run-paths dataclass
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class RunPaths:
    run_dir: Path
    config_path: Path
    sweep_path: Path
    stdout_path: Path
    stderr_path: Path
    csv_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.csv_path = self.run_dir / ENGINE_CSV_NAME


# ─────────────────────────────────────────────────────────────────────────
# State helpers
# ─────────────────────────────────────────────────────────────────────────
def ensure_state() -> None:
    """Populate ``st.session_state`` with safe defaults on first render."""
    if "config" not in st.session_state:
        st.session_state.config = json.loads(json.dumps(DEFAULT_CONFIG))
    if "sweep_rows" not in st.session_state:
        st.session_state.sweep_rows = [
            {
                "enabled": True,
                "parameter": parameter,
                "values": ", ".join(
                    str(v) if not isinstance(v, (dict, list)) else json.dumps(v)
                    for v in values
                ),
            }
            for parameter, values in DEFAULT_SWEEP.items()
        ]
    if "last_run_dir" not in st.session_state:
        st.session_state.last_run_dir = None
    if "last_returncode" not in st.session_state:
        st.session_state.last_returncode = None
    if "last_result_df" not in st.session_state:
        st.session_state.last_result_df = None
    if "execution_mode" not in st.session_state:
        st.session_state.execution_mode = "Single Point"
    if "single_point_result" not in st.session_state:
        st.session_state.single_point_result = None
    if "dwdm_enabled" not in st.session_state:
        st.session_state.dwdm_enabled = False
    if "auto_optimize_pulses" not in st.session_state:
        st.session_state.auto_optimize_pulses = False


# ─────────────────────────────────────────────────────────────────────────
# Scalar & list parsing helpers
# ─────────────────────────────────────────────────────────────────────────
def parse_scalar(raw: str) -> Any:
    """Parse a single user-typed scalar string into a Python value.

    Handles ``true``/``false``/``none``, JSON-encoded dicts/lists, ints,
    floats (including scientific notation), and falls back to a raw string.
    """
    value = raw.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None
    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_sweep_value_list(raw: str) -> list[Any]:
    """Parse a comma-separated or JSON-list sweep value string."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("Sweep value must be a list.")
        return parsed
    values: list[Any] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(parse_scalar(part))
    return values


def build_sweep_from_rows(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert the editable sweep table into a ``{dotted_path: [values]}`` dict."""
    sweep: dict[str, list[Any]] = {}
    for row in rows:
        if not row.get("enabled", True):
            continue
        parameter = str(row.get("parameter", "")).strip()
        values_raw = str(row.get("values", "")).strip()
        if not parameter or not values_raw:
            continue
        values = parse_sweep_value_list(values_raw)
        if values:
            sweep[parameter] = values
    return sweep


def count_combinations(sweep: dict[str, list[Any]]) -> int:
    total = 1
    for values in sweep.values():
        total *= max(len(values), 1)
    return total


# ─────────────────────────────────────────────────────────────────────────
# Nested-dict traversal (used to read defaults into the UI)
# ─────────────────────────────────────────────────────────────────────────
def get_nested_value(config: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Resolve ``a.b.c`` against a nested dict; return ``default`` if missing."""
    current: Any = config
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def set_nested_value(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set ``a.b.c`` on a nested dict, creating intermediate dicts as needed."""
    keys = dotted_path.split(".")
    current: Any = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


# ─────────────────────────────────────────────────────────────────────────
# Filesystem helpers
# ─────────────────────────────────────────────────────────────────────────
def create_run_paths(base_output_dir: str) -> RunPaths:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir).expanduser().resolve() / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        sweep_path=run_dir / "sweep.json",
        stdout_path=run_dir / "stdout.log",
        stderr_path=run_dir / "stderr.log",
    )


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def find_result_files(run_dir: Path) -> list[Path]:
    patterns = ["*.csv", "*.json", "**/*.csv", "**/*.json"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(run_dir.glob(pattern))
    excluded = {"config.json", "sweep.json"}
    unique_files = sorted(
        {path.resolve() for path in files if path.name not in excluded},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return unique_files


def load_result_dataframe(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".json":
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            if isinstance(raw, dict):
                for key in ("results", "data", "records", "simulations"):
                    if isinstance(raw.get(key), list):
                        return pd.DataFrame(raw[key])
                return pd.json_normalize(raw)
    except Exception as exc:
        st.warning(f"Cannot load result file `{path.name}`: {exc}")
    return None


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [str(col).strip() for col in normalized.columns]
    return normalized


def guess_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for column in columns:
            if candidate_lower in column.lower():
                return column
    return None


# ─────────────────────────────────────────────────────────────────────────
# Validation — emits inline warnings without blocking the UI
# ─────────────────────────────────────────────────────────────────────────
def validate_config(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Return a list of (severity, message) tuples for spec-mandated constraints.

    Severities: ``"warning"``, ``"error"``.  Errors should be treated as
    blocking before run; warnings are advisory.
    """
    findings: list[tuple[str, str]] = []
    sim = config.get("simulation", {})
    proto = config.get("protocol_runtime", {})
    protocol_params = config.get("protocol_params", {})
    detector = config.get("detector", {})
    source = config.get("source", {})
    pulses = source.get("pulses", {})
    epsilons = config.get("protocol", {}).get("epsilons", {})

    # ── min ≤ max pulses ────────────────────────────────────────────────────
    min_log = sim.get("min_pulses_log")
    max_log = sim.get("max_pulses_log")
    if min_log is not None and max_log is not None and min_log > max_log:
        findings.append(("error", f"min_pulses_log ({min_log}) > max_pulses_log ({max_log})."))

    # ── distance_points ≥ 1 ──────────────────────────────────────────────────
    dp = sim.get("distance_points")
    if dp is not None and dp < 1:
        findings.append(("error", f"distance_points ({dp}) must be ≥ 1."))

    # ── distance_start ≤ distance_stop ──────────────────────────────────────
    ds = sim.get("distance_start_km")
    dt = sim.get("distance_stop_km")
    if ds is not None and dt is not None and ds > dt:
        findings.append(("error", f"distance_start_km ({ds}) > distance_stop_km ({dt})."))

    # ── P_signal + P_decoy + P_vacuum = 1 ────────────────────────────────────
    try:
        total_prob = sum(p.get("prob", 0.0) for p in pulses.values())
        if abs(total_prob - 1.0) > 0.005:
            findings.append(("warning", f"Pulse probabilities sum to {total_prob:.3f} (should be 1.0). Engine will auto-balance."))
    except Exception:
        pass

    # ── epsilons positive & sum < 1 ─────────────────────────────────────────
    eps_keys = ("eps_sec", "eps_cor", "eps_pe", "eps_smooth", "eps_pa", "eps_phase_est")
    eps_values = [epsilons.get(k, 0.0) for k in eps_keys]
    if any(v <= 0 for v in eps_values):
        findings.append(("error", "All ε values must be strictly positive."))
    eps_sum = sum(eps_values)
    if eps_sum >= 1.0:
        findings.append(("error", f"Σ ε = {eps_sum:.2e} ≥ 1.0 (must be < 1.0)."))

    # ── MA2005/WANG2005 secondary constraint (eps_pe + eps_cor + eps_smooth + eps_pa ≤ eps_sec) ─
    proof = source.get("intended_proof", "LIM_2014")
    if proof in ("MA_2005", "WANG_2005"):
        secondary = epsilons.get("eps_pe", 0) + epsilons.get("eps_cor", 0) + epsilons.get("eps_smooth", 0) + epsilons.get("eps_pa", 0)
        if secondary > epsilons.get("eps_sec", 0):
            findings.append(("warning", f"{proof}: ε_pe + ε_cor + ε_smooth + ε_pa = {secondary:.2e} > ε_sec; engine will auto-rescale to ε_sec/10."))

    # ── bias_voltage > breakdown_voltage (SPD only) ──────────────────────────
    det_type = detector.get("detector_type", "SPD")
    if det_type == "SPD":
        bv = detector.get("bias_voltage", 0)
        bdv = detector.get("breakdown_voltage", 0)
        if bv <= bdv:
            findings.append(("error", f"bias_voltage ({bv} V) ≤ breakdown_voltage ({bdv} V) for SPD — Geiger mode guard will fail."))

    # ── PNRD + strict_mode + (dead_time/afterpulse/jitter > 0) ──────────────
    if det_type == "PNRD" and detector.get("strict_mode", False):
        if (detector.get("dead_time_ns", 0) > 0
                or detector.get("afterpulse_prob", 0) > 0
                or detector.get("jitter_fwhm_ns", 0) > 0):
            findings.append(("warning", "PNRD + strict_mode + non-zero dead_time/afterpulse/jitter will hard-fail inside _validate_params()."))

    # ── rogers_2007 + dead_time_ns == 0 ─────────────────────────────────────
    if detector.get("double_click_policy") == "rogers_2007" and detector.get("dead_time_ns", 0) <= 0:
        findings.append(("warning", "rogers_2007 double-click policy is a no-op when dead_time_ns == 0 (analytic SBR returns None)."))

    # ── THERMAL + ADVERSARIAL_BLOCK warning ─────────────────────────────────
    if source.get("statistics_type") == "THERMAL" and source.get("error_model") == "ADVERSARIAL_BLOCK":
        findings.append(("warning", "THERMAL + ADVERSARIAL_BLOCK may produce unexpected results."))

    # ── f_EC ≥ 1.0 ──────────────────────────────────────────────────────────
    f_ec = protocol_params.get("error_correction_efficiency", 1.16)
    if f_ec < 1.0:
        findings.append(("error", f"error_correction_efficiency ({f_ec}) must be ≥ 1.0."))

    # ── z-basis probabilities in [0,1] ──────────────────────────────────────
    for k in ("alice_z_basis_prob", "bob_z_basis_prob", "z_basis_prob"):
        v = proto.get(k)
        if v is not None and not (0.0 <= v <= 1.0):
            findings.append(("warning", f"{k} = {v} is outside [0, 1]."))

    # ── parameter_estimation_fraction in (0, 1) ─────────────────────────────
    pe_frac = proto.get("parameter_estimation_fraction")
    if pe_frac is not None and not (0.0 < pe_frac < 1.0):
        findings.append(("warning", f"parameter_estimation_fraction = {pe_frac} should be in (0, 1)."))

    return findings


def render_findings(findings: list[tuple[str, str]]) -> bool:
    """Render validation findings inline; return True if any errors are present."""
    has_errors = any(sev == "error" for sev, _ in findings)
    if not findings:
        return False
    for sev, msg in findings:
        if sev == "error":
            st.error(f"⛔ {msg}")
        else:
            st.warning(f"⚠️ {msg}")
    return has_errors


# ─────────────────────────────────────────────────────────────────────────
# Academic Plotly theme
# ─────────────────────────────────────────────────────────────────────────
ACADEMIC_LAYOUT: dict[str, Any] = dict(
    template="plotly_white",
    font=dict(family="Latin Modern Roman, Computer Modern, DejaVu Serif, serif", size=14, color="#222"),
    title=dict(font=dict(size=17, family="Latin Modern Roman, serif"), x=0.5, xanchor="center"),
    xaxis=dict(showgrid=True, gridcolor="#ddd", linecolor="#444", mirror=True, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#ddd", linecolor="#444", mirror=True, zeroline=False),
    legend=dict(bordercolor="#888", borderwidth=1, bgcolor="rgba(255,255,255,0.85)"),
    margin=dict(l=72, r=24, t=64, b=64),
    hovermode="x unified",
)


def apply_academic_layout(fig: go.Figure, x_label: str = "", y_label: str = "",
                          log_x: bool = False, log_y: bool = False) -> go.Figure:
    """Apply the academic theme + axis labels to a Plotly figure."""
    layout_updates = dict(ACADEMIC_LAYOUT)
    xaxis: dict[str, Any] = dict(layout_updates.pop("xaxis", {}))
    yaxis: dict[str, Any] = dict(layout_updates.pop("yaxis", {}))
    if log_x:
        xaxis.update(type="log")
    if log_y:
        yaxis.update(type="log")
    if x_label:
        xaxis.update(title=dict(text=x_label, standoff=12))
    if y_label:
        yaxis.update(title=dict(text=y_label, standoff=12))
    fig.update_layout(**layout_updates)
    fig.update_xaxes(**xaxis)
    fig.update_yaxes(**yaxis)
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Image export (PNG / PDF) via kaleido
# ─────────────────────────────────────────────────────────────────────────
def export_plotly_image(fig: go.Figure, fmt: str, width: int = 1200, height: int = 700,
                        scale: float = 2.0) -> bytes:
    """Render a Plotly figure to bytes (PNG or PDF).

    Requires the optional ``kaleido`` package.  Raises with a friendly
    message if kaleido is missing.
    """
    fmt = fmt.lower()
    if fmt not in ("png", "pdf", "svg", "eps", "webp"):
        raise ValueError(f"Unsupported image format: {fmt}")
    try:
        return fig.to_image(format=fmt, width=width, height=height, scale=scale)
    except Exception as exc:
        raise RuntimeError(
            f"Plotly image export failed (is `kaleido` installed? `pip install kaleido`). Underlying error: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────
def init_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.45rem; }
        div[data-testid="stMetricLabel"] { font-size: 0.95rem; color: #555; }
        .small-muted { color: #666; font-size: 0.88rem; }
        .katex { font-size: 1.05em; }
        /* Tighter Plotly captions */
        .stCaption { font-style: italic; color: #555; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if _ENGINE_IMPORT_ERROR is not None:
        st.error(
            f"⚠️ Engine import failed — UI is running in dry-run mode. "
            f"Reason: `{_ENGINE_IMPORT_ERROR}`. Install `main_optimized` + `qkd.*` "
            f"to enable actual simulation runs."
        )


def render_header() -> None:
    st.title("🔐 QKD Simulation Dashboard")
    st.caption(APP_SUBTITLE)


# ─────────────────────────────────────────────────────────────────────────
# Sidebar — Execution settings
# ─────────────────────────────────────────────────────────────────────────
def render_sidebar() -> tuple[str, int, str, bool, bool, int | None, float | None, float | None]:
    """Render execution-mode controls in the sidebar; return user selections.

    Returns ``(output_dir, workers, mode, dwdm, auto_optimize,
    wdm_channel_count, wdm_channel_power_dbm, wdm_raman_coefficient)``.
    """
    st.sidebar.header("Execution")
    mode = st.sidebar.radio(
        "Run mode",
        options=["Single Point", "1-D Sweep", "2-D Sweep"],
        index=0,
        help=(
            "**Single Point**: collapse the grid to one (distance, pulses) pair — "
            "fast feedback + KPI cards. **1-D Sweep**: classic sweep over one "
            "parameter at multiple distances. **2-D Sweep**: contour over "
            "(distance, second_param)."
        ),
        key="execution_mode",
    )

    output_dir = st.sidebar.text_input(
        "Output directory", value=DEFAULT_OUTPUT_DIR,
        help="Each run gets its own subfolder under this path.",
    )
    workers = st.sidebar.number_input(
        "Workers",
        min_value=1,
        max_value=max(os.cpu_count() or 1, 1),
        value=max((os.cpu_count() or 2) - 1, 1),
        step=1,
        help=f"Number of parallel processes. Your system has {os.cpu_count()} CPUs.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("DWDM / Raman mode")
    dwdm = st.sidebar.checkbox(
        "Enable DWDM (--dwdm)",
        value=False,
        help="Toggle 4-channel WDM preset (Raman noise + AWG filter loss).",
    )
    wdm_channel_count: int | None = None
    wdm_channel_power_dbm: float | None = None
    wdm_raman_coefficient: float | None = None
    if dwdm:
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            # NOTE: inside a sidebar column context, use `st.number_input` (no
            # `st.sidebar.` prefix) — otherwise the widget escapes the column.
            wdm_channel_count = st.number_input(
                "WDM channel count", min_value=1, max_value=64, value=4, step=1,
                key="dwdm_channel_count",
            )
            wdm_raman_coefficient = st.number_input(
                "Raman coefficient", value=1e-9, format="%.2e",
                help="Raman scattering coefficient (Hz range).",
                key="dwdm_raman_coefficient",
            )
        with col_b:
            paper_faithful = st.checkbox("Paper-faithful power (None)", value=True,
                                         key="dwdm_paper_faithful")
            if not paper_faithful:
                wdm_channel_power_dbm = st.number_input(
                    "Per-channel launch power (dBm)",
                    value=-20.0, min_value=-40.0, max_value=10.0, step=1.0,
                    key="dwdm_channel_power_dbm",
                )

    st.sidebar.divider()
    st.sidebar.subheader("Auto-optimisation")
    auto_optimize = st.sidebar.checkbox(
        "Auto-optimize pulses (--auto-optimize-pulses)",
        value=False,
        help=(
            "Run per-(combo, distance) SLSQP optimisation of (μ, ν, p_s, p_d, p_v). "
            "Cached across noise=False/True rows for fair comparison."
        ),
        key="auto_optimize_pulses",
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"*Engine: {'✅ loaded' if _ENGINE_IMPORT_ERROR is None else '⚠️ missing'}*  "
        f"\n*MAX_DISTANCE_KM = {MAX_DISTANCE_KM}*"
    )

    return (output_dir, int(workers), mode, dwdm, auto_optimize,
            wdm_channel_count, wdm_channel_power_dbm, wdm_raman_coefficient)


# ─────────────────────────────────────────────────────────────────────────
# Config editor — complete parameter parity with SIMULATION_SPEC.md
# ─────────────────────────────────────────────────────────────────────────
def render_config_editor() -> dict[str, Any]:
    """Render the full parameter surface across five categorised tabs."""
    st.subheader("Base Configuration")
    st.caption("Every parameter in SIMULATION_SPEC.md §2 is exposed below. Hover tooltips quote the verbatim spec default.")

    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    tab_chan, tab_det, tab_src, tab_proto, tab_sim = st.tabs([
        "🌐 Channel", "🔎 Detector", "💡 Source", "🧩 Protocol", "⚙️ Simulation"
    ])

    # ==================== CHANNEL TAB (spec §2.1.1) ====================
    with tab_chan:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            config["channel"]["fiber_loss_db_km"] = st.number_input(
                "Fibre loss α  (dB/km)",
                min_value=0.0, max_value=float(MAX_FIBER_LOSS_DB_KM),
                value=float(config["channel"]["fiber_loss_db_km"]),
                step=0.01, format="%.3f",
                help="Standard SMF-28 at 1550 nm: 0.20 dB/km.",
            )
        with col_c2:
            config["channel"]["dispersion_parameter_ps_nm_km"] = st.number_input(
                "Chromatic dispersion D  (ps/nm/km)",
                min_value=0.0, max_value=100.0,
                value=float(config["channel"]["dispersion_parameter_ps_nm_km"]),
                step=0.1, format="%.2f",
                help="Standard SMF at 1550 nm ≈ 17 ps/nm/km. Set 0 to disable.",
            )

    # ==================== DETECTOR TAB (spec §2.1.2 + §2.1.3) ============
    with tab_det:
        st.markdown("#### Basic Parameters")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            config["detector"]["det_eff_d0"] = st.number_input(
                "η_D0  (dimensionless)",
                min_value=0.0, max_value=1.0,
                value=float(config["detector"]["det_eff_d0"]),
                step=0.01, format="%.3f",
                help="Quantum efficiency of detector D0.",
            )
            config["detector"]["dark_rate"] = st.number_input(
                "Dark count rate D0  (Hz)",
                min_value=0.0, max_value=1e6,
                value=float(config["detector"]["dark_rate"]),
                step=1e-7, format="%.2e",
                help="SPAD ≈ 600 Hz; SNSPD ≈ 1 Hz; PNRD ≈ 1000 Hz.",
            )
        with col_d2:
            config["detector"]["det_eff_d1"] = st.number_input(
                "η_D1  (dimensionless)",
                min_value=0.0, max_value=1.0,
                value=float(config["detector"]["det_eff_d1"]),
                step=0.01, format="%.3f",
            )
            dark_d1_val = st.number_input(
                "Dark count rate D1  (Hz, 0 → same as D0)",
                min_value=0.0, max_value=1e6,
                value=float(config["detector"]["dark_rate_d1"] or 0.0),
                step=1e-7, format="%.2e",
                help="Set to 0 to mirror `dark_rate` (engine semantics).",
            )
            config["detector"]["dark_rate_d1"] = dark_d1_val if dark_d1_val > 0 else None
        with col_d3:
            config["detector"]["qber_intrinsic"] = st.number_input(
                "Intrinsic QBER e_det  (dimensionless)",
                min_value=0.0, max_value=0.1,
                value=float(config["detector"]["qber_intrinsic"]),
                step=0.001, format="%.4f",
                help="Baseline detector+optics QBER (typical 0.005).",
            )
            config["detector"]["misalignment"] = st.number_input(
                "Misalignment e_mis  (dimensionless)",
                min_value=0.0, max_value=0.05,
                value=float(config["detector"]["misalignment"]),
                step=0.002, format="%.4f",
            )

        st.markdown("#### Detector Type & Statistical Models")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            config["detector"]["detector_type"] = st.selectbox(
                "Detector family", options=["SPD", "PNRD", "SNSPD"], index=0,
                help="Triggers DETECTOR_PRESETS cascade (efficiency, dark rate, jitter, ...).",
            )
            config["detector"]["dead_time_model"] = st.selectbox(
                "Dead-time model",
                options=["NON_PARALYZABLE", "PARALYZABLE"],
                index=0,
                help="PARALYZABLE is deprecated (F-28).",
            )
        with col_m2:
            config["detector"]["afterpulse_model"] = st.selectbox(
                "Afterpulse model", options=["EXPONENTIAL", "GEOMETRIC"], index=0,
            )
            config["detector"]["strict_mode"] = st.checkbox(
                "Strict mode",
                value=bool(config["detector"]["strict_mode"]),
                help="Raise on incompatible PNRD + dead_time/afterpulse/jitter > 0.",
            )
        with col_m3:
            config["detector"]["double_click_policy"] = st.selectbox(
                "Double-click policy (detector-level)",
                options=["RANDOM", "DISCARD", "rogers_2007"],
                index=0,
                help="`rogers_2007` requires dead_time_ns > 0.",
            )
            config["protocol_params"]["detector_type"] = config["detector"]["detector_type"]

        st.markdown("#### Timing Parameters")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            config["detector"]["dead_time_ns"] = st.number_input(
                "Dead time τ_d  (ns)",
                min_value=0.0, max_value=1000.0,
                value=float(config["detector"]["dead_time_ns"]),
                step=1.0, format="%.2f",
            )
        with col_t2:
            config["detector"]["jitter_fwhm_ns"] = st.number_input(
                "Jitter FWHM σ_t  (ns)",
                min_value=0.0, max_value=1.0,
                value=float(config["detector"]["jitter_fwhm_ns"]),
                step=0.01, format="%.4f",
                help="SNSPD ≈ 0.020 ns; SPAD ≈ 0.300 ns.",
            )
        with col_t3:
            config["detector"]["afterpulse_prob"] = st.number_input(
                "Afterpulse probability p_ap  (dimensionless)",
                min_value=0.0, max_value=0.05,
                value=float(config["detector"]["afterpulse_prob"]),
                step=0.001, format="%.4f",
            )
            config["detector"]["afterpulse_lifetime_ns"] = st.number_input(
                "Afterpulse lifetime τ_ap  (ns)",
                min_value=1.0, max_value=10000.0,
                value=float(config["detector"]["afterpulse_lifetime_ns"]),
                step=10.0, format="%.2f",
            )

        with st.expander("🌡️ Temperature & Bias Voltage (Dark-count scaling)"):
            st.caption("`bias_voltage ≤ breakdown_voltage` → 0.0 dark rate (Geiger-mode guard).")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                config["detector"]["temperature_k"] = st.number_input(
                    "Operating temperature T  (K)", min_value=0.0, max_value=400.0,
                    value=float(config["detector"]["temperature_k"]), step=1.0,
                )
                config["detector"]["ref_temperature_k"] = st.number_input(
                    "Reference temperature T_ref  (K)", min_value=0.0, max_value=400.0,
                    value=float(config["detector"]["ref_temperature_k"]), step=1.0,
                )
            with col_v2:
                config["detector"]["bias_voltage"] = st.number_input(
                    "Bias voltage V_bias  (V)", min_value=0.0, max_value=100.0,
                    value=float(config["detector"]["bias_voltage"]), step=0.5,
                    help="⚠️ If ≤ breakdown_voltage, dark rate → 0.",
                )
                config["detector"]["breakdown_voltage"] = st.number_input(
                    "Breakdown voltage V_br  (V)", min_value=0.0, max_value=100.0,
                    value=float(config["detector"]["breakdown_voltage"]), step=0.5,
                )
                config["detector"]["ref_bias_voltage"] = st.number_input(
                    "Reference bias V_ref  (V)", min_value=0.0, max_value=100.0,
                    value=float(config["detector"]["ref_bias_voltage"]), step=0.5,
                )

    # ==================== SOURCE TAB (spec §2.2.2 + §2.2.3) ==============
    with tab_src:
        st.markdown("#### Decoy-state pulse intensities (3-pulse scheme)")
        col_s, col_d, col_v = st.columns(3)
        with col_s:
            config["source"]["pulses"]["signal"]["mu"] = st.number_input(
                "Signal μ_s  (photons/pulse)",
                min_value=0.0, max_value=2.0,
                value=float(config["source"]["pulses"]["signal"]["mu"]),
                step=0.01, format="%.3f",
            )
            config["source"]["pulses"]["signal"]["prob"] = st.number_input(
                "P(signal)", min_value=0.0, max_value=1.0,
                value=float(config["source"]["pulses"]["signal"]["prob"]),
                step=0.01, format="%.3f",
            )
        with col_d:
            config["source"]["pulses"]["decoy"]["mu"] = st.number_input(
                "Decoy μ_d  (photons/pulse)",
                min_value=0.0, max_value=2.0,
                value=float(config["source"]["pulses"]["decoy"]["mu"]),
                step=0.01, format="%.3f",
            )
            config["source"]["pulses"]["decoy"]["prob"] = st.number_input(
                "P(decoy)", min_value=0.0, max_value=1.0,
                value=float(config["source"]["pulses"]["decoy"]["prob"]),
                step=0.01, format="%.3f",
            )
        with col_v:
            config["source"]["pulses"]["vacuum"]["mu"] = st.number_input(
                "Vacuum μ_v  (photons/pulse)",
                min_value=0.0, max_value=0.01,
                value=float(config["source"]["pulses"]["vacuum"]["mu"]),
                step=0.001, format="%.4f",
                help="True vacuum = 0.0.",
            )
            config["source"]["pulses"]["vacuum"]["prob"] = st.number_input(
                "P(vacuum)", min_value=0.0, max_value=1.0,
                value=float(config["source"]["pulses"]["vacuum"]["prob"]),
                step=0.01, format="%.3f",
            )

        total_prob = sum(p["prob"] for p in config["source"]["pulses"].values())
        if abs(total_prob - 1.0) > 0.005:
            st.warning(f"P(signal) + P(decoy) + P(vacuum) = {total_prob:.3f} ≠ 1.0 — engine will auto-balance.")

        st.markdown("#### Source Properties")
        col_sp1, col_sp2, col_sp3 = st.columns(3)
        with col_sp1:
            config["source"]["pulse_period_ns"] = st.number_input(
                "Pulse period T_p  (ns)", min_value=0.1, max_value=1000.0,
                value=float(config["source"]["pulse_period_ns"]),
                step=0.1, format="%.2f",
                help="10 ns → 100 MHz source rate.",
            )
            config["source"]["intensity_jitter"] = st.number_input(
                "Intensity jitter", min_value=0.0, max_value=1.0,
                value=float(config["source"]["intensity_jitter"]),
                step=0.01, format="%.3f",
            )
        with col_sp2:
            config["source"]["modulation_index"] = st.number_input(
                "Modulation index", min_value=0.0, max_value=1.0,
                value=float(config["source"]["modulation_index"]),
                step=0.01, format="%.3f",
            )
            config["source"]["extinction_ratio"] = st.number_input(
                "Extinction ratio  (dB)", min_value=0.0, max_value=120.0,
                value=float(config["source"]["extinction_ratio"]),
                step=1.0, format="%.1f",
            )
        with col_sp3:
            config["source"]["ideal_emission_probability"] = st.number_input(
                "P(ideal emission)", min_value=0.0, max_value=1.0,
                value=float(config["source"]["ideal_emission_probability"]),
                step=0.01, format="%.3f",
            )
            config["source"]["source_fidelity"] = st.number_input(
                "Source fidelity F", min_value=0.0, max_value=1.0,
                value=float(config["source"]["source_fidelity"]),
                step=0.001, format="%.4f",
            )

        st.markdown("#### Statistics & Error Model")
        col_se1, col_se2, col_se3 = st.columns(3)
        with col_se1:
            config["source"]["statistics_type"] = st.selectbox(
                "Statistics type", options=["POISSON", "THERMAL"], index=0,
            )
        with col_se2:
            config["source"]["error_model"] = st.selectbox(
                "Error model",
                options=["RANDOM_GAUSSIAN", "ADVERSARIAL_BLOCK"],
                index=0,
            )
        with col_se3:
            config["source"]["adversarial_block_size"] = st.number_input(
                "Adversarial block size N_b  (pulses)",
                min_value=1, max_value=10_000_000,
                value=int(config["source"]["adversarial_block_size"]),
                step=100,
            )

        with st.expander("🔗 Channel & Approximation Toggles"):
            col_cc1, col_cc2 = st.columns(2)
            with col_cc1:
                config["source"]["N_channels"] = st.number_input(
                    "N_channels (WDM)", min_value=1, max_value=64,
                    value=int(config["source"]["N_channels"]), step=1,
                )
                config["source"]["is_bidirectional"] = st.checkbox(
                    "Bidirectional fibre",
                    value=bool(config["source"]["is_bidirectional"]),
                )
            with col_cc2:
                config["source"]["preferred_lp_solver"] = st.selectbox(
                    "LP solver", options=["highs", "scipy", "cvxopt"], index=0,
                )
                config["source"]["expected_decoder"] = st.selectbox(
                    "Expected decoder", options=["LOCAL", "ENTANGLING"], index=0,
                )
                config["source"]["assumed_double_click"] = st.selectbox(
                    "Assumed double click",
                    options=["RANDOM", "DISCARD"], index=0,
                )

            config["source"]["use_small_angle_approximation"] = st.checkbox(
                "Use small-angle approximation",
                value=bool(config["source"]["use_small_angle_approximation"]),
            )
            config["source"]["use_linear_modulation_approximation"] = st.checkbox(
                "Use linear-modulation approximation",
                value=bool(config["source"]["use_linear_modulation_approximation"]),
            )

        with st.expander("📡 MZM Modulator"):
            mzm = config["source"]["mzm"]
            col_mzm1, col_mzm2 = st.columns(2)
            with col_mzm1:
                mzm["extinction_ratio_db"] = st.number_input(
                    "MZM extinction ratio  (dB)", min_value=0.0, max_value=120.0,
                    value=float(mzm["extinction_ratio_db"]), step=1.0, format="%.1f",
                )
                mzm["v_pi"] = st.number_input(
                    "MZM V_π  (V)", min_value=0.1, max_value=20.0,
                    value=float(mzm["v_pi"]), step=0.1, format="%.2f",
                )
            with col_mzm2:
                mzm["bias_voltage"] = st.number_input(
                    "MZM bias voltage  (V)", min_value=0.0, max_value=20.0,
                    value=float(mzm["bias_voltage"]), step=0.1, format="%.2f",
                )
                mzm["rf_amplitude_v"] = st.number_input(
                    "MZM RF amplitude  (V)", min_value=0.0, max_value=20.0,
                    value=float(mzm["rf_amplitude_v"]), step=0.1, format="%.2f",
                )

        with st.expander("⚡ Electrical Noise (Driver)"):
            en = config["source"]["electrical_noise"]
            col_en1, col_en2 = st.columns(2)
            with col_en1:
                en["voltage_std_v"] = st.number_input(
                    "Voltage σ_v  (V)", min_value=0.0, max_value=1.0,
                    value=float(en["voltage_std_v"]), step=0.01, format="%.4f",
                )
            with col_en2:
                en["bandwidth_hz"] = st.number_input(
                    "Noise bandwidth  (Hz)", min_value=1e3, max_value=1e12,
                    value=float(en["bandwidth_hz"]), step=1e8, format="%.2e",
                )

        with st.expander("🌡️ Source Physical Parameters"):
            col_pp1, col_pp2, col_pp3 = st.columns(3)
            with col_pp1:
                config["source"]["temperature_k"] = st.number_input(
                    "Source temperature  (K)", min_value=0.0, max_value=500.0,
                    value=float(config["source"]["temperature_k"]), step=1.0,
                )
                config["source"]["v_pi"] = st.number_input(
                    "Source V_π  (V)", min_value=0.1, max_value=20.0,
                    value=float(config["source"]["v_pi"]), step=0.1, format="%.2f",
                )
            with col_pp2:
                config["source"]["bandwidth_hz"] = st.number_input(
                    "Source bandwidth  (Hz)", min_value=1e3, max_value=1e12,
                    value=float(config["source"]["bandwidth_hz"]),
                    step=1e8, format="%.2e",
                )
                config["source"]["phi_bias"] = st.number_input(
                    "φ_bias  (rad)", min_value=0.0, max_value=2 * math.pi,
                    value=float(config["source"]["phi_bias"]),
                    step=0.01, format="%.4f",
                )
            with col_pp3:
                config["source"]["driver_load_resistance_ohm"] = st.number_input(
                    "Driver R_load  (Ω)", min_value=1.0, max_value=10000.0,
                    value=float(config["source"]["driver_load_resistance_ohm"]),
                    step=10.0, format="%.1f",
                )
                config["source"]["dc_photocurrent_a"] = st.number_input(
                    "DC photocurrent  (A)", min_value=0.0, max_value=1e-3,
                    value=float(config["source"]["dc_photocurrent_a"]),
                    step=1e-7, format="%.2e",
                )
                config["source"]["max_intensity_mu"] = st.number_input(
                    "Max intensity μ_cap  (photons/pulse)",
                    min_value=0.5, max_value=10.0,
                    value=float(config["source"]["max_intensity_mu"]),
                    step=0.5, format="%.2f",
                )

        with st.expander("🛡️ Security-Proof Settings"):
            col_sec1, col_sec2 = st.columns(2)
            with col_sec1:
                config["source"]["intended_proof"] = st.selectbox(
                    "Intended proof",
                    options=["LIM_2014", "MA_2005", "WANG_2005", "TIGHT"],
                    index=0,
                    help="MA2005/WANG2005 trigger epsilon auto-rescale.",
                )
            with col_sec2:
                config["source"]["confidence_method"] = st.selectbox(
                    "Confidence-bound method",
                    options=["GAUSSIAN", "HOEFFDING", "CLOPPER_PEARSON"],
                    index=0,
                )

        with st.expander("🧬 Density Matrices (Advanced)"):
            st.caption("Provide custom density matrices per pulse type; leave empty to disable.")
            dm_str = st.text_area(
                "Density matrices (JSON or empty)",
                value="",
                height=150,
                help='Example: {"signal": [[1.0, 0.0], [0.0, 0.0]], "decoy": [[1.0, 0.0], [0.0, 0.0]], "vacuum": [[1.0, 0.0], [0.0, 0.0]]}',
            )
            if dm_str.strip():
                try:
                    config["source"]["density_matrices"] = json.loads(dm_str)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON for density matrices: {e}")
                    config["source"]["density_matrices"] = None
            else:
                config["source"]["density_matrices"] = None

        with st.expander("🔐 Source Security Metadata (Advanced)"):
            sm_str = st.text_area(
                "Security metadata (JSON or empty)", value="", height=100,
            )
            if sm_str.strip():
                try:
                    config["source"]["security_metadata"] = json.loads(sm_str)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON for security metadata: {e}")
                    config["source"]["security_metadata"] = None
            else:
                config["source"]["security_metadata"] = None

    # ==================== PROTOCOL TAB (spec §2.2.1, §2.2.4, §2.2.5) ====
    with tab_proto:
        st.markdown("#### Protocol Family & Runtime")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            config["protocol_params"]["protocol"] = st.selectbox(
                "Protocol family",
                options=["BB84_DECOY", "B92", "MDI_QKD", "REDUNDANT"],
                index=0,
            )
            config["protocol_runtime"]["protocol_class"] = st.selectbox(
                "Protocol class (Python)",
                options=[
                    "BB84DecoyProtocol",
                    "B92Protocol",
                    "MDIQKDProtocol",
                    "RedundantTransmissionProtocol",
                ],
                index=0,
            )
            config["protocol_params"]["error_correction_efficiency"] = st.number_input(
                "Error-correction efficiency f_EC",
                min_value=1.0, max_value=3.0,
                value=float(config["protocol_params"]["error_correction_efficiency"]),
                step=0.01, format="%.3f",
                help="Cascade/LDPC inefficiency. Ideal=1.0; typical=1.16.",
            )
        with col_p2:
            config["protocol_runtime"]["parameter_estimation_fraction"] = st.number_input(
                "Parameter-estimation fraction",
                min_value=0.01, max_value=0.5,
                value=float(config["protocol_runtime"]["parameter_estimation_fraction"]),
                step=0.01, format="%.3f",
            )
            config["protocol_runtime"]["num_worker_rngs"] = st.number_input(
                "Num worker RNGs",
                min_value=1, max_value=16,
                value=int(config["protocol_runtime"]["num_worker_rngs"]),
                step=1,
                help="Hard-coded to 5 inside run_single_simulation.",
            )

        with st.expander("📐 Basis Probabilities"):
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                config["protocol_runtime"]["alice_z_basis_prob"] = st.number_input(
                    "P(Alice chooses Z)", min_value=0.0, max_value=1.0,
                    value=float(config["protocol_runtime"]["alice_z_basis_prob"]),
                    step=0.01, format="%.3f",
                )
            with col_b2:
                config["protocol_runtime"]["bob_z_basis_prob"] = st.number_input(
                    "P(Bob chooses Z)", min_value=0.0, max_value=1.0,
                    value=float(config["protocol_runtime"]["bob_z_basis_prob"]),
                    step=0.01, format="%.3f",
                )
            with col_b3:
                config["protocol_runtime"]["z_basis_prob"] = st.number_input(
                    "P(Z, MDI symmetric)", min_value=0.0, max_value=1.0,
                    value=float(config["protocol_runtime"]["z_basis_prob"]),
                    step=0.01, format="%.3f",
                )

        with st.expander("👆 Double-Click Policies"):
            col_dc1, col_dc2 = st.columns(2)
            with col_dc1:
                config["protocol_runtime"]["double_click_policy"] = st.selectbox(
                    "Protocol double-click policy",
                    options=["DISCARD", "RANDOM", "rogers_2007"], index=0,
                )
            with col_dc2:
                st.caption("Detector-level policy is set on the Detector tab.")

        with st.expander("🧲 Entangling Decoder (RedundantTransmissionProtocol only)"):
            config["protocol_runtime"]["use_entangling_decoder"] = st.checkbox(
                "Use entangling decoder",
                value=bool(config["protocol_runtime"]["use_entangling_decoder"]),
            )
            config["protocol_runtime"]["use_entangling_encoder"] = st.checkbox(
                "Use entangling encoder",
                value=bool(config["protocol_runtime"]["use_entangling_encoder"]),
            )
            config["protocol_runtime"]["redundancy_M"] = st.number_input(
                "Redundancy M", min_value=1, max_value=10,
                value=int(config["protocol_runtime"]["redundancy_M"]), step=1,
            )
            codeword_str = st.text_input(
                "Codeword mapping (JSON or empty)", value="",
                help="Leave empty for None, or provide JSON dict.",
            )
            config["protocol_runtime"]["codeword_mapping"] = (
                json.loads(codeword_str) if codeword_str.strip() else None
            )
            damping_val = st.number_input(
                "Damping parameter γ", min_value=0.0, max_value=1.0,
                value=float(config["protocol_runtime"].get("damping_parameter") or 0.0),
                step=0.01, format="%.3f",
            )
            config["protocol_runtime"]["damping_parameter"] = damping_val if damping_val > 0 else None
            rot_str = st.text_input(
                "Rotation angle θ  (radians or empty)", value="",
                help="Leave empty for None.",
            )
            config["protocol_runtime"]["rotation_angle"] = float(rot_str) if rot_str.strip() else None

        with st.expander("🔐 Security Epsilons (ALL 6)"):
            eps = config["protocol"]["epsilons"]
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                eps["eps_sec"] = st.number_input("ε_sec",      value=float(eps["eps_sec"]),      format="%.2e")
                eps["eps_cor"] = st.number_input("ε_cor",      value=float(eps["eps_cor"]),      format="%.2e")
            with col_e2:
                eps["eps_pe"]     = st.number_input("ε_pe",     value=float(eps["eps_pe"]),     format="%.2e")
                eps["eps_smooth"] = st.number_input("ε_smooth", value=float(eps["eps_smooth"]), format="%.2e")
            with col_e3:
                eps["eps_pa"]       = st.number_input("ε_pa",       value=float(eps["eps_pa"]),       format="%.2e")
                eps["eps_phase_est"] = st.number_input("ε_phase_est", value=float(eps["eps_phase_est"]), format="%.2e")
            eps_sum = sum(eps.values())
            st.caption(f"Σ ε = {eps_sum:.2e}  (must be < 1.0)")

        with st.expander("📜 Security Certificate"):
            config["protocol"]["assumed_phase_equals_bit_error"] = st.checkbox(
                "Assume e_phase = e_bit (conservative)",
                value=bool(config["protocol"].get("assumed_phase_equals_bit_error", True)),
                help="F-16: conservative fallback for phase-error estimation.",
            )

    # ==================== SIMULATION TAB (spec §2.3.1) ==================
    with tab_sim:
        st.info(
            "ℹ️ `main_optimized` runs **multiple** pulse counts (min_log → max_log) "
            "and distances per sweep combo, with both `noise=False` and `noise=True` rows."
        )
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            config["simulation"]["min_pulses_log"] = st.number_input(
                "log10(min pulses)", min_value=3, max_value=14,
                value=int(config["simulation"]["min_pulses_log"]), step=1,
                help="10^4 = 10 000 pulses. Must be ≤ max.",
            )
            config["simulation"]["max_pulses_log"] = st.number_input(
                "log10(max pulses)", min_value=3, max_value=14,
                value=int(config["simulation"]["max_pulses_log"]), step=1,
                help="10^12 approaches the asymptotic regime.",
            )
        with col_s2:
            config["simulation"]["distance_start_km"] = st.number_input(
                "Distance start  (km)", min_value=0.0, max_value=float(MAX_DISTANCE_KM),
                value=float(config["simulation"]["distance_start_km"]), step=1.0,
            )
            config["simulation"]["distance_stop_km"] = st.number_input(
                "Distance stop  (km)", min_value=0.0, max_value=float(MAX_DISTANCE_KM),
                value=float(config["simulation"]["distance_stop_km"]), step=10.0,
            )
            config["simulation"]["distance_points"] = st.number_input(
                "Distance points N_d",
                min_value=1, max_value=1000,
                value=int(config["simulation"]["distance_points"]),
                step=1,
            )

        col_sim1, col_sim2, col_sim3 = st.columns(3)
        with col_sim1:
            config["simulation"]["rng_seed"] = st.number_input(
                "RNG seed",
                value=int(config["simulation"]["rng_seed"]),
                step=1,
                help="Master seed for SeedSequence.spawn(5).",
            )
        with col_sim2:
            config["simulation"]["apply_statistical_noise"] = st.checkbox(
                "Apply statistical noise",
                value=bool(config["simulation"]["apply_statistical_noise"]),
                help="Master toggle for noise=True rows.",
            )
        with col_sim3:
            config["simulation"]["sampling_cap_per_pulse_type"] = st.number_input(
                "Sampling cap / pulse type",
                min_value=10_000, max_value=10_000_000,
                value=int(config["simulation"]["sampling_cap_per_pulse_type"]),
                step=10_000,
            )

        n_pulse_counts = max(0, config["simulation"]["max_pulses_log"] - config["simulation"]["min_pulses_log"] + 1)
        n_distances = config["simulation"]["distance_points"]
        noise_multiplier = 2 if config["simulation"]["apply_statistical_noise"] else 1
        st.info(
            f"Per sweep combo: **{n_pulse_counts}** pulse counts × **{noise_multiplier}** "
            f"noise settings × **{n_distances}** distances = "
            f"**{n_pulse_counts * noise_multiplier * n_distances}** simulations."
        )

        with st.expander("🔧 Local Optimisation (Ma 2005 / Chapman 2018)"):
            config["simulation"]["enable_optimization"] = st.checkbox(
                "Apply local optimisation before sweep",
                value=bool(config["simulation"].get("enable_optimization", False)),
                help=(
                    "Pre-sweep: optimise (μ, ν) via `optimize_decoy_intensities` "
                    "(and `calculate_optimal_mu_ma2005` for MA_2005), and rotation "
                    "angle via `optimize_rotation_angle_chapman` for "
                    "RedundantTransmissionProtocol."
                ),
            )
            if config["simulation"]["enable_optimization"]:
                proof = config["source"].get("intended_proof", "LIM_2014")
                if proof == "MA_2005":
                    st.info("→ Will use `calculate_optimal_mu_ma2005` + `optimize_decoy_intensities`.")
                else:
                    st.info(f"→ Will use `optimize_decoy_intensities` (proof: {proof}).")
                if config["protocol_runtime"]["protocol_class"] == "RedundantTransmissionProtocol":
                    st.info("→ Will also use `optimize_rotation_angle_chapman`.")

    # Validation findings (rendered inline at the bottom of the Setup tab).
    # We do NOT call st.stop() here because that would prevent the other tabs
    # (Execute, Results, Files, Help) from rendering.  The Execute panel
    # re-validates and refuses to launch if any error-severity findings exist.
    findings = validate_config(config)
    has_errors = render_findings(findings)
    st.session_state["_config_validation_errors"] = has_errors

    # Raw JSON editor (override-all)
    with st.expander("📝 Raw JSON Editor (Override All)"):
        raw = st.text_area("Edit JSON", value=json.dumps(config, indent=2), height=400)
        try:
            config = json.loads(raw)
            st.success("Valid JSON.")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

    return config


# ─────────────────────────────────────────────────────────────────────────
# Sweep editor — 1-D / 2-D mode-aware
# ─────────────────────────────────────────────────────────────────────────
def render_sweep_editor(execution_mode: str) -> dict[str, list[Any]]:
    """Render the sweep editor. In Single Point mode, returns an empty dict."""
    st.subheader("Parameter Sweep")
    if execution_mode == "Single Point":
        st.info(
            "🎯 **Single Point mode** — the sweep grid is collapsed to one "
            "(distance, pulses) pair. Configure distance & pulses on the "
            "Simulation tab; results will appear as KPI cards on the Execute tab."
        )
        return {}

    st.caption("Use dotted paths for nested parameters, e.g. `channel.fiber_loss_db_km` or `detector.det_eff_d0`.")

    # ── Mode-specific quick-load buttons ───────────────────────────────
    st.markdown(f"#### {execution_mode} quick-load presets")
    col_q1, col_q2, col_q3 = st.columns(3)
    with col_q1:
        if st.button("Load 1-D preset (μ_s × distance)", key="preset_1d"):
            st.session_state.sweep_rows = [
                {"enabled": True, "parameter": "source.pulses.signal.mu",
                 "values": "0.3, 0.5, 0.8"},
            ]
            st.rerun()
    with col_q2:
        if st.button("Load 2-D preset (μ_s × det_eff_d0)", key="preset_2d"):
            st.session_state.sweep_rows = [
                {"enabled": True, "parameter": "source.pulses.signal.mu",
                 "values": "0.3, 0.5, 0.8"},
                {"enabled": True, "parameter": "detector.det_eff_d0",
                 "values": "0.10, 0.15, 0.20"},
            ]
            st.rerun()
    with col_q3:
        if st.button("Clear all", key="clear_sweep"):
            st.session_state.sweep_rows = []
            st.rerun()

    # ── Add parameter from catalog ─────────────────────────────────────
    with st.expander("➕ Add Parameter from Catalog", expanded=False):
        groups: dict[str, list[str]] = {}
        for param, info in MASTER_SWEEP_CATALOG.items():
            groups.setdefault(info["group"], []).append(param)

        col_add1, col_add2 = st.columns([2, 1])
        with col_add1:
            selected_group = st.selectbox(
                "Category", options=list(groups.keys()),
                key="cat_group",
            )
            selected_param = st.selectbox(
                "Parameter", options=groups[selected_group],
                format_func=lambda x: x.replace(".", " → "),
                key="cat_param",
            )

        # Compute the default values string for the currently-selected param.
        default_vals = MASTER_SWEEP_CATALOG[selected_param]["default"]
        default_str = ", ".join(
            str(v) if not isinstance(v, (dict, list)) else json.dumps(v)
            for v in default_vals
        )

        # ── Render-time sentinel: if the selected parameter has changed
        # since the last render, OVERWRITE the cached "cat_values" widget
        # value with the freshly-computed default. This is Streamlit's
        # officially-recommended pattern for programmatically setting a
        # widget value mid-script — see the docs at
        # https://docs.streamlit.io/library-api-reference/session-state
        #
        # The pop-and-rely-on-value= pattern doesn't reliably reinitialise
        # the widget on all Streamlit versions; direct assignment is more
        # bulletproof. We ALSO assign on the very first render (when
        # cat_values is not yet in session_state) to seed the initial value.
        _last_cat_param = st.session_state.get("_last_cat_param")
        _param_changed = (
            _last_cat_param is not None
            and _last_cat_param != selected_param
        )
        _first_render = "cat_values" not in st.session_state
        if _param_changed or _first_render:
            st.session_state["cat_values"] = default_str
        st.session_state["_last_cat_param"] = selected_param

        with col_add2:
            # NOTE: we intentionally do NOT pass value= here — Streamlit
            # raises StreamlitAPIException if a widget's key is already
            # set in session_state AND value= is also passed. The widget
            # will use session_state["cat_values"] which we just set.
            new_values_str = st.text_area(
                "Values", height=80,
                help="Comma-separated or JSON list. (Defaults are auto-loaded when you switch Parameter.)",
                key="cat_values",
            )
        if st.button("Add to Sweep", type="primary", key="add_to_sweep"):
            existing_params = [row.get("parameter", "") for row in st.session_state.sweep_rows]
            if selected_param in existing_params:
                st.warning(f"Parameter '{selected_param}' already exists.")
            else:
                st.session_state.sweep_rows.append({
                    "enabled": True,
                    "parameter": selected_param,
                    "values": new_values_str,
                })
                # After a successful add, wipe the cached Values widget so the
                # next pick starts from a clean default (users typically add
                # several params in a row).
                st.session_state.pop("cat_values", None)
                st.rerun()

    # ── Add custom parameter ────────────────────────────────────────────
    with st.expander("✏️ Add Custom Parameter", expanded=False):
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            custom_param = st.text_input("Custom dotted path", placeholder="e.g. detector.det_eff_d0", key="cust_path")
        with col_c2:
            custom_values = st.text_input("Values", placeholder="0.1, 0.2, 0.3", key="cust_vals")
        if st.button("Add Custom", key="add_custom"):
            if custom_param and custom_values:
                st.session_state.sweep_rows.append({
                    "enabled": True, "parameter": custom_param, "values": custom_values,
                })
                st.rerun()

    st.markdown("---")

    # ── Editable table ──────────────────────────────────────────────────
    edited_rows = st.data_editor(
        st.session_state.sweep_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled", default=True),
            "parameter": st.column_config.TextColumn("Parameter", help="e.g. channel.fiber_loss_db_km"),
            "values": st.column_config.TextColumn("Values", help="e.g. 0, 25, 50 or [0, 25, 50]"),
        },
        key="sweep_data_editor",
    )
    st.session_state.sweep_rows = list(edited_rows)

    try:
        sweep = build_sweep_from_rows(st.session_state.sweep_rows)
        combinations = count_combinations(sweep)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Sweep parameters", len(sweep))
        with col_m2:
            st.metric("Total combinations", f"{combinations:,}")
        if combinations > 10_000:
            st.warning("Large combination count — simulation may take a long time.")
        if combinations > 100_000:
            st.error("🚨 > 100 000 combinations — this may take hours.")
        with st.expander("Sweep JSON preview"):
            st.json(sweep)
        active_params = set(sweep.keys())
        missing_params = set(MASTER_SWEEP_CATALOG.keys()) - active_params
        if missing_params:
            with st.expander(f"Available but not active ({len(missing_params)} parameters)"):
                for group in ["Source", "Detector", "Channel", "Protocol", "Epsilon"]:
                    group_params = [p for p in missing_params if MASTER_SWEEP_CATALOG[p]["group"] == group]
                    if group_params:
                        st.markdown(f"**{group}:**")
                        for p in group_params:
                            st.text(f"  • {p}")
        return sweep
    except Exception as exc:
        st.error(f"Sweep parsing error: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────
# Local optimisation (applied before sweep if enabled)
# ─────────────────────────────────────────────────────────────────────────
def _apply_optimization_to_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply local optimisation (Ma 2005 / Chapman 2018) at the midpoint distance."""
    sim = config["simulation"]
    dist_start = float(sim.get("distance_start_km", 0.0))
    dist_stop = float(sim.get("distance_stop_km", 150.0))
    rep_distance = (dist_start + dist_stop) / 2.0 if dist_stop > dist_start else dist_start

    fiber_loss = float(config["channel"]["fiber_loss_db_km"])
    det_eff = float(config["detector"]["det_eff_d0"])
    dark_rate = float(config["detector"]["dark_rate"])
    f_ec = float(config["protocol_params"]["error_correction_efficiency"])
    proof = config["source"].get("intended_proof", "LIM_2014")

    if proof == "MA_2005":
        e_detector = float(config["detector"].get("qber_intrinsic", 0.01)) + float(
            config["detector"].get("misalignment", 0.005)
        )
        mu_opt = calculate_optimal_mu_ma2005(
            distance_km=rep_distance, fiber_loss_db_km=fiber_loss, det_eff=det_eff,
            dark_rate=dark_rate, error_correction_f=f_ec, e_detector=e_detector,
        )
        _, nu_opt = optimize_decoy_intensities(
            distance_km=rep_distance, fiber_loss_db_km=fiber_loss, det_eff=det_eff,
            dark_rate=dark_rate, error_correction_f=f_ec,
        )
    else:
        mu_opt, nu_opt = optimize_decoy_intensities(
            distance_km=rep_distance, fiber_loss_db_km=fiber_loss, det_eff=det_eff,
            dark_rate=dark_rate, error_correction_f=f_ec,
        )
    config["source"]["pulses"]["signal"]["mu"] = mu_opt
    config["source"]["pulses"]["decoy"]["mu"] = nu_opt

    if config["protocol_runtime"]["protocol_class"] == "RedundantTransmissionProtocol":
        gamma = float(config["protocol_runtime"].get("damping_parameter") or 0.0)
        M = int(config["protocol_runtime"].get("redundancy_M", 2))
        theta_opt = optimize_rotation_angle_chapman(gamma=gamma, M=M)
        config["protocol_runtime"]["rotation_angle"] = theta_opt

    return config


# ─────────────────────────────────────────────────────────────────────────
# Engine-call wrapper (forwards DWDM & auto_optimize kwargs)
# ─────────────────────────────────────────────────────────────────────────
def _invoke_engine(
    config: dict[str, Any],
    workers: int,
    sweep: dict[str, list[Any]],
    run_dir: Path,
    stdout_collector: io.StringIO,
    dwdm: bool = False,
    wdm_channel_count: int | None = None,
    wdm_channel_power_dbm: float | None = None,
    wdm_raman_coefficient: float | None = None,
    auto_optimize_pulses: bool = False,
) -> None:
    """Invoke the engine in ``run_dir`` with stdout redirected to a collector."""
    prev_cwd = os.getcwd()
    kwargs: dict[str, Any] = dict(
        config=config, num_workers=workers, sweeps=sweep,
    )
    if dwdm:
        kwargs["dwdm"] = True
        if wdm_channel_count is not None:
            kwargs["wdm_channel_count"] = wdm_channel_count
        if wdm_channel_power_dbm is not None:
            kwargs["wdm_channel_power_dbm"] = wdm_channel_power_dbm
        if wdm_raman_coefficient is not None:
            kwargs["wdm_raman_coefficient"] = wdm_raman_coefficient
    if auto_optimize_pulses:
        kwargs["auto_optimize_pulses"] = True
    try:
        os.chdir(run_dir)
        with contextlib.redirect_stdout(stdout_collector):
            run_and_save_csv(**kwargs)
    finally:
        os.chdir(prev_cwd)


# ─────────────────────────────────────────────────────────────────────────
# Single Point run — cached, returns the single-row DataFrame
# ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running single-point simulation…", max_entries=8, ttl=3600)
def _cached_single_point_run(
    config_json: str,
    workers: int,
    dwdm: bool,
    wdm_channel_count: int | None,
    wdm_channel_power_dbm: float | None,
    wdm_raman_coefficient: float | None,
    auto_optimize_pulses: bool,
    output_dir: str,
) -> tuple[str, str]:
    """Cached single-point run.

    Cache key is the FULL input tuple — ``run_dir`` is NOT a parameter; it is
    derived deterministically from the input hash so that cache hits return the
    same on-disk directory (and the same CSV).

    Returns ``(csv_text, run_dir_str)``.
    """
    import hashlib
    key_payload = json.dumps([
        config_json, workers, dwdm, wdm_channel_count,
        wdm_channel_power_dbm, wdm_raman_coefficient, auto_optimize_pulses,
    ], sort_keys=True)
    digest = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:16]
    run_dir = Path(output_dir).expanduser().resolve() / f"single_point_cache_{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_json)
    sweep: dict[str, list[Any]] = {}
    collector = io.StringIO()
    _invoke_engine(
        config=config, workers=workers, sweep=sweep, run_dir=run_dir,
        stdout_collector=collector, dwdm=dwdm,
        wdm_channel_count=wdm_channel_count,
        wdm_channel_power_dbm=wdm_channel_power_dbm,
        wdm_raman_coefficient=wdm_raman_coefficient,
        auto_optimize_pulses=auto_optimize_pulses,
    )
    csv_path = run_dir / ENGINE_CSV_NAME
    csv_text = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
    return (csv_text, str(run_dir))


# ─────────────────────────────────────────────────────────────────────────
# Execute panel — mode-aware (single / sweep)
# ─────────────────────────────────────────────────────────────────────────
def render_execute_panel(
    config: dict[str, Any],
    sweep: dict[str, list[Any]],
    output_dir: str,
    workers: int,
    mode: str,
    dwdm: bool,
    auto_optimize: bool,
    wdm_channel_count: int | None = None,
    wdm_channel_power_dbm: float | None = None,
    wdm_raman_coefficient: float | None = None,
) -> None:
    st.subheader("Run Simulation")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Mode", mode)
    with col_b:
        st.metric("Combinations", count_combinations(sweep) if sweep else 1)
    with col_c:
        st.metric("Workers", workers)
    with col_d:
        st.metric("Output", output_dir)

    # ── Local optimisation pre-pass ─────────────────────────────────────
    if config.get("simulation", {}).get("enable_optimization", False):
        config = _apply_optimization_to_config(config)

    run_clicked = st.button("🚀 Run simulation", type="primary", use_container_width=True)
    if not run_clicked:
        return

    # ── Validate before launching ───────────────────────────────────────
    findings = validate_config(config)
    if any(sev == "error" for sev, _ in findings):
        st.error("⛔ Configuration has blocking errors — fix them on the Setup tab before running.")
        for sev, msg in findings:
            if sev == "error":
                st.error(f"⛔ {msg}")
        return

    paths = create_run_paths(output_dir)
    write_json(paths.config_path, config)
    write_json(paths.sweep_path, sweep if sweep else {})
    st.session_state.last_run_dir = str(paths.run_dir)
    st.session_state.last_returncode = None
    st.info(f"Run directory: `{paths.run_dir}`")

    # ── Mode dispatch ───────────────────────────────────────────────────
    if mode == "Single Point":
        _run_single_point(config, workers, paths, output_dir,
                          dwdm, auto_optimize,
                          wdm_channel_count, wdm_channel_power_dbm, wdm_raman_coefficient)
    else:
        _run_sweep(config, workers, sweep, paths, dwdm, auto_optimize,
                   wdm_channel_count, wdm_channel_power_dbm, wdm_raman_coefficient)


def _run_single_point(
    config: dict[str, Any], workers: int, paths: RunPaths,
    output_dir: str,
    dwdm: bool, auto_optimize: bool,
    wdm_channel_count: int | None,
    wdm_channel_power_dbm: float | None,
    wdm_raman_coefficient: float | None,
) -> None:
    """Run a single-point simulation and render KPI cards."""
    # Collapse the grid to a single point: 1 distance, single pulse count.
    single_config = json.loads(json.dumps(config))
    single_config["simulation"]["distance_points"] = 1
    single_config["simulation"]["max_pulses_log"] = single_config["simulation"]["min_pulses_log"]
    single_config["simulation"]["apply_statistical_noise"] = True

    progress = st.progress(0, text="Launching single-point run…")
    status_box = st.empty()
    captured_lines: list[str] = []

    class _LineCollector(io.StringIO):
        def write(self, text: str) -> int:
            if text.strip():
                captured_lines.append(text.rstrip())
            return super().write(text)

    collector = _LineCollector()
    start_time = time.time()

    try:
        with st.spinner("Running engine…"):
            csv_text, cached_run_dir_str = _cached_single_point_run(
                config_json=json.dumps(single_config, sort_keys=True),
                workers=workers,
                dwdm=dwdm, wdm_channel_count=wdm_channel_count,
                wdm_channel_power_dbm=wdm_channel_power_dbm,
                wdm_raman_coefficient=wdm_raman_coefficient,
                auto_optimize_pulses=auto_optimize,
                output_dir=output_dir,
            )
        # Replace the freshly-created (empty) run_dir with the cached one so
        # downstream file-browser / Results Explorer can find the CSV.
        st.session_state.last_run_dir = cached_run_dir_str
        progress.progress(1.0, text="Done.")
        st.session_state.last_returncode = 0
        status_box.success("Single-point run complete (cached).")
    except Exception as exc:
        progress.progress(1.0, text="Failed.")
        st.session_state.last_returncode = 1
        status_box.error(f"Single-point run failed: {exc}")
        return
    finally:
        try:
            paths.stdout_path.write_text(collector.getvalue(), encoding="utf-8")
        except Exception:
            pass

    # ── Parse the single-row CSV into KPI cards ─────────────────────────
    if not csv_text.strip():
        st.warning("Engine produced no CSV output.")
        return
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        st.error(f"Failed to parse engine output: {exc}")
        return
    if df.empty:
        st.warning("Engine produced an empty CSV.")
        return
    st.session_state.last_result_df = df
    _render_kpi_cards(df)
    elapsed = time.time() - start_time
    st.info(f"Elapsed: {elapsed:.1f}s")


def _render_kpi_cards(df: pd.DataFrame) -> None:
    """Render KPI summary cards for a single-row (or single-point) DataFrame."""
    if df.empty:
        return
    row = df.iloc[-1]  # prefer the noise=True row when both present

    def _fmt(v: Any, fmt: str = "") -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            try:
                return fmt.format(float(v)) if fmt else f"{float(v):.4g}"
            except Exception:
                return str(v)
        return str(v)

    st.markdown("#### KPI Summary")
    cols = st.columns(4)
    metrics = [
        ("Secure Key Rate (bits/pulse)", "secure_key_rate", "%.4e"),
        ("Secure Key Bits",              "secure_key_bits", "%d"),
        ("QBER",                         "qber",            "%.4e"),
        ("Channel Loss (dB)",            "channel_total_loss_db", "%.3f"),
        ("Transmittance",                "channel_transmittance", "%.3e"),
        ("Sifted Bits",                  "raw_sifted_bits", "%d"),
        ("Detection Yield (signal)",     "detection_yield_signal", "%.4e"),
        ("Simulation Time (s)",          "simulation_time_sec", "%.2f"),
    ]
    for i, (label, col_name, fmt) in enumerate(metrics):
        if col_name in df.columns:
            cols[i % 4].metric(label, _fmt(row.get(col_name), fmt))


def _run_sweep(
    config: dict[str, Any], workers: int, sweep: dict[str, list[Any]], paths: RunPaths,
    dwdm: bool, auto_optimize: bool,
    wdm_channel_count: int | None,
    wdm_channel_power_dbm: float | None,
    wdm_raman_coefficient: float | None,
) -> None:
    """Run a sweep simulation with a live progress bar."""
    progress = st.progress(0, text="Launching sweep…")
    status_box = st.empty()
    log_box = st.empty()
    captured_lines: list[str] = []

    class _LineCollector(io.StringIO):
        def write(self, text: str) -> int:
            if text.strip():
                captured_lines.append(text.rstrip())
            return super().write(text)

    collector = _LineCollector()
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _invoke_engine,
            config=config, workers=workers, sweep=sweep, run_dir=paths.run_dir,
            stdout_collector=collector, dwdm=dwdm,
            wdm_channel_count=wdm_channel_count,
            wdm_channel_power_dbm=wdm_channel_power_dbm,
            wdm_raman_coefficient=wdm_raman_coefficient,
            auto_optimize_pulses=auto_optimize,
        )
        while not future.done():
            elapsed = time.time() - start_time
            progress.progress(min(0.95, elapsed / max(elapsed + 30.0, 1.0)),
                              text=f"Running sweep… elapsed: {elapsed:.1f}s")
            status_box.info(f"Running… elapsed: {elapsed:.1f}s")
            if captured_lines:
                log_box.code("\n".join(captured_lines[-80:]), language="text")
            time.sleep(0.5)

    try:
        future.result()
        progress.progress(1.0, text="Done.")
        st.session_state.last_returncode = 0
        status_box.success("Sweep finished successfully.")
    except Exception as exc:
        progress.progress(1.0, text="Failed.")
        st.session_state.last_returncode = 1
        status_box.error(f"Sweep failed: {exc}")

    try:
        paths.stdout_path.write_text(collector.getvalue(), encoding="utf-8")
    except Exception:
        pass

    if paths.csv_path.exists():
        st.info(f"Results: `{paths.csv_path}`")
        try:
            df = pd.read_csv(paths.csv_path)
            st.session_state.last_result_df = df
        except Exception:
            pass
    elapsed = time.time() - start_time
    st.info(f"Elapsed: {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────
# Plotly figure builders — academic styling, units, log scales
# ─────────────────────────────────────────────────────────────────────────
def _filter_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to OK rows with positive SKR for plotting."""
    if "status" in df.columns:
        df = df[df["status"] == "OK"]
    if "secure_key_rate" in df.columns:
        df = df[df["secure_key_rate"] > 0]
    return df


def fig_skr_vs_distance(df: pd.DataFrame) -> go.Figure | None:
    """Spec §4.2.1 — Secure Key Rate vs Distance, log Y, noise series."""
    required = {"distance_km", "secure_key_rate"}
    if not required.issubset(df.columns):
        return None
    plot_df = _filter_plot_df(df)
    if plot_df.empty:
        return None
    fig = go.Figure()
    noise_col = "noise_applied" in plot_df.columns
    pulses_col = "total_pulses" in plot_df.columns
    # Two lines per (combo) — noiseless (False) and noisy (True)
    if noise_col:
        group_cols = ["noise_applied"]
        if pulses_col:
            group_cols.append("total_pulses")
        for grp_keys, sub in plot_df.groupby(group_cols, sort=False):
            if not isinstance(grp_keys, tuple):
                grp_keys = (grp_keys,)
            noise_on = bool(grp_keys[0])
            line_style = dict(dash="solid" if noise_on else "dash")
            parts = [f"noise={'ON' if noise_on else 'OFF'}"]
            if pulses_col:
                parts.append(f"N_pulses={grp_keys[1]:.0e}")
            label = " | ".join(parts)
            sub_sorted = sub.sort_values("distance_km")
            fig.add_trace(go.Scatter(
                x=sub_sorted["distance_km"], y=sub_sorted["secure_key_rate"],
                mode="lines+markers", name=label, line=line_style,
                connectgaps=True,
            ))
    else:
        sub_sorted = plot_df.sort_values("distance_km")
        fig.add_trace(go.Scatter(
            x=sub_sorted["distance_km"], y=sub_sorted["secure_key_rate"],
            mode="lines+markers", name="SKR",
        ))
    # Rogers cross-check overlay (optional, spec §4.2.1)
    if "analytic_rogers_sbr" in plot_df.columns:
        rogers = plot_df.dropna(subset=["analytic_rogers_sbr"]).sort_values("distance_km")
        if not rogers.empty:
            fig.add_trace(go.Scatter(
                x=rogers["distance_km"], y=rogers["analytic_rogers_sbr"],
                mode="markers", name="Rogers (2007) Eq. 15",
                marker=dict(symbol="x", color="black", size=8),
            ))
    apply_academic_layout(
        fig, x_label="Distance L  (km)", y_label="Secure Key Rate  (bits/pulse, log10)",
        log_y=True,
    )
    fig.update_layout(title="Secure Key Rate vs Distance")
    return fig


def fig_qber_vs_distance(df: pd.DataFrame) -> go.Figure | None:
    """Spec §4.2.2 — QBER vs Distance, with 11% threshold and CI band."""
    if not {"distance_km", "qber"}.issubset(df.columns):
        return None
    plot_df = df.copy()
    if "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"] == "OK"]
    plot_df = plot_df.sort_values("distance_km")
    fig = go.Figure()
    # CI band if both bounds present
    if {"qber_ci_low", "qber_ci_high"}.issubset(plot_df.columns):
        fig.add_trace(go.Scatter(
            x=plot_df["distance_km"], y=plot_df["qber_ci_high"],
            mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=plot_df["distance_km"], y=plot_df["qber_ci_low"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(100,100,100,0.2)", hoverinfo="skip",
            name="95% CI",
        ))
    fig.add_trace(go.Scatter(
        x=plot_df["distance_km"], y=plot_df["qber"],
        mode="lines+markers", name="QBER (overall)",
    ))
    # Per-pulse-type QBER
    for col, label in (("qber_signal", "QBER (signal)"),
                       ("qber_decoy", "QBER (decoy)"),
                       ("qber_vacuum", "QBER (vacuum)")):
        if col in plot_df.columns:
            sub = plot_df.dropna(subset=[col])
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["distance_km"], y=sub[col],
                    mode="lines", name=label,
                ))
    # 11% threshold
    fig.add_hline(y=0.11, line_dash="dash", line_color="red",
                  annotation_text="BB84 abort threshold (11%)",
                  annotation_position="top left")
    apply_academic_layout(
        fig, x_label="Distance L  (km)", y_label="QBER  (dimensionless, log10)",
        log_y=True,
    )
    fig.update_layout(title="QBER vs Distance")
    return fig


def fig_yield_vs_distance(df: pd.DataFrame) -> go.Figure | None:
    """Spec §4.2.3 — Detection Yield vs Distance, log Y, three series."""
    yield_cols = ["detection_yield_signal", "detection_yield_decoy", "detection_yield_vacuum"]
    if not {"distance_km", *yield_cols}.issubset(df.columns):
        return None
    plot_df = df.copy()
    if "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"] == "OK"]
    plot_df = plot_df.sort_values("distance_km")
    fig = go.Figure()
    for col, label, color in zip(
        yield_cols,
        ["Q_signal", "Q_decoy", "Q_vacuum"],
        ["#1f77b4", "#ff7f0e", "#2ca02c"],
    ):
        sub = plot_df.dropna(subset=[col])
        sub = sub[sub[col] > 0]
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=sub["distance_km"], y=sub[col],
                mode="lines+markers", name=label, line=dict(color=color),
            ))
    apply_academic_layout(
        fig, x_label="Distance L  (km)", y_label="Detection Yield Q_μ  (dimensionless, log10)",
        log_y=True,
    )
    fig.update_layout(title="Detection Yield vs Distance")
    return fig


def fig_loss_budget(df: pd.DataFrame) -> go.Figure | None:
    """Spec §4.2.4 — Channel Loss Budget (DWDM twin-axis)."""
    if "channel_total_loss_db" not in df.columns:
        return None
    plot_df = df.copy().sort_values("distance_km")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["distance_km"], y=plot_df["channel_total_loss_db"],
        mode="lines+markers", name="Total loss (dB)",
        line=dict(color="#1f77b4"),
    ))
    if "params_raman_dark_rate_hz" in plot_df.columns:
        raman = plot_df.dropna(subset=["params_raman_dark_rate_hz"])
        raman = raman[raman["params_raman_dark_rate_hz"] > 0]
        if not raman.empty:
            fig.add_trace(go.Scatter(
                x=raman["distance_km"], y=raman["params_raman_dark_rate_hz"],
                mode="lines+markers", name="Raman dark rate (Hz)",
                yaxis="y2", line=dict(color="#ff7f0e", dash="dot"),
            ))
            fig.update_layout(yaxis2=dict(
                title="Raman dark rate  (Hz, log10)", overlaying="y", side="right",
                type="log", showgrid=False,
            ))
    apply_academic_layout(
        fig, x_label="Distance L  (km)", y_label="Channel loss  (dB)",
    )
    fig.update_layout(title="Channel Loss Budget (DWDM)")
    return fig


def fig_finite_size_contour(df: pd.DataFrame) -> go.Figure | None:
    """Spec §4.2.5 — Finite-size contour: SKR vs (Distance, Block size)."""
    required = {"distance_km", "total_pulses", "secure_key_rate"}
    if not required.issubset(df.columns):
        return None
    plot_df = df.copy()
    if "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"] == "OK"]
    plot_df = plot_df[plot_df["secure_key_rate"] > 0]
    if plot_df.empty:
        return None
    fig = go.Figure(data=go.Heatmap(
        x=plot_df["distance_km"],
        y=np.log10(plot_df["total_pulses"]),
        z=np.log10(plot_df["secure_key_rate"]),
        colorscale="Viridis",
        colorbar=dict(title="log10 SKR"),
        hovertemplate="L=%{x:.1f} km<br>log10(N)=%{y:.2f}<br>log10(SKR)=%{z:.2f}<extra></extra>",
    ))
    apply_academic_layout(
        fig, x_label="Distance L  (km)", y_label="log10(N_pulses)",
    )
    fig.update_layout(title="Finite-Size Contour — Secure Key Rate")
    return fig


def fig_sweep_sensitivity(df: pd.DataFrame) -> go.Figure | None:
    """Spec §4.2.6 — Sweep sensitivity: SKR vs swept parameter at fixed (L, N)."""
    sweep_cols = [c for c in df.columns if c.startswith("sweep_")]
    if not sweep_cols or "secure_key_rate" not in df.columns:
        return None
    sweep_col = sweep_cols[0]
    plot_df = df.copy()
    if "status" in plot_df.columns:
        plot_df = plot_df[plot_df["status"] == "OK"]
    plot_df = plot_df[plot_df["secure_key_rate"] > 0]
    if plot_df.empty:
        return None
    # Pick the largest total_pulses for a clean single line
    if "total_pulses" in plot_df.columns:
        max_pulses = plot_df["total_pulses"].max()
        plot_df = plot_df[plot_df["total_pulses"] == max_pulses]
    # Pick the distance closest to the median for a representative cross-section
    if "distance_km" in plot_df.columns and not plot_df.empty:
        median_d = plot_df["distance_km"].median()
        closest_dist = plot_df.iloc[(plot_df["distance_km"] - median_d).abs().argsort()[:1]]["distance_km"].iloc[0]
        plot_df = plot_df[plot_df["distance_km"] == closest_dist]
    plot_df = plot_df.sort_values(sweep_col)
    if plot_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df[sweep_col], y=plot_df["secure_key_rate"],
        mode="lines+markers", name="SKR",
    ))
    # Argmax marker
    idx = plot_df["secure_key_rate"].idxmax()
    x_star = plot_df.loc[idx, sweep_col]
    fig.add_vline(x=x_star, line_dash="dash", line_color="red",
                  annotation_text=f"argmax SKR @ {x_star:.3g}")
    apply_academic_layout(
        fig,
        x_label=sweep_col.replace("sweep_", "").replace("_", " → "),
        y_label="Secure Key Rate  (bits/pulse, log10)",
        log_y=True,
    )
    fig.update_layout(title=f"Sweep Sensitivity — SKR vs {sweep_col}")
    return fig


def fig_convergence_audit(df: pd.DataFrame) -> go.Figure | None:
    """Spec §4.2.7 — Convergence audit: simulation time and failure rate."""
    if "simulation_time_sec" not in df.columns:
        return None
    plot_df = df.copy()
    if "protocol_class" not in plot_df.columns:
        plot_df["protocol_class"] = "Unknown"
    if "total_pulses" not in plot_df.columns:
        plot_df["total_pulses"] = 1
    grouped = (
        plot_df.groupby(["protocol_class", "total_pulses"])["simulation_time_sec"]
        .agg(["mean", "std"]).reset_index()
    )
    fig = go.Figure()
    for proto, sub in grouped.groupby("protocol_class"):
        sub = sub.sort_values("total_pulses")
        fig.add_trace(go.Bar(
            x=sub["total_pulses"].astype(str),
            y=sub["mean"],
            error_y=dict(array=sub["std"].fillna(0), type="data"),
            name=str(proto),
        ))
    apply_academic_layout(
        fig, x_label="Total Pulses (log10 scale)", y_label="Wall time (s)",
    )
    fig.update_layout(title="Convergence Audit — Per-Row Wall Time", barmode="group")
    fig.update_xaxes(type="log")
    return fig


# ─────────────────────────────────────────────────────────────────────────
# Results explorer
# ─────────────────────────────────────────────────────────────────────────
def render_results_panel() -> None:
    st.subheader("Results Explorer")

    default_dir = st.session_state.last_run_dir or ""
    selected_run_dir = st.text_input(
        "Run directory", value=default_dir,
        help="Path to the run output directory (containing the engine CSV).",
    )
    if not selected_run_dir:
        st.info("No run directory yet — run a simulation on the Execute tab.")
        return

    run_dir = Path(selected_run_dir).expanduser()
    if not run_dir.exists():
        st.error("Run directory does not exist.")
        return

    # Prefer the canonical engine CSV if present; otherwise list all CSVs.
    csv_path = run_dir / ENGINE_CSV_NAME
    if csv_path.exists():
        selected_file = csv_path
    else:
        result_files = find_result_files(run_dir)
        if not result_files:
            st.warning("No CSV/JSON result files in this directory.")
            return
        selected_file = st.selectbox(
            "Result file", options=result_files,
            format_func=lambda p: str(p.relative_to(run_dir)) if p.is_relative_to(run_dir) else str(p),
        )

    df = load_result_dataframe(selected_file)
    if df is None or df.empty:
        st.warning("The selected file has no displayable data.")
        return
    df = normalize_dataframe_columns(df)
    st.session_state.last_result_df = df

    # ── Filters ──────────────────────────────────────────────────────────
    st.markdown("#### QKD Data Filters")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        if "status" in df.columns:
            valid_statuses = sorted(df["status"].astype(str).unique().tolist())
            selected_status = st.multiselect("Status", options=valid_statuses, default=valid_statuses)
            df = df[df["status"].astype(str).isin(selected_status)]
    with col_f2:
        if "noise_applied" in df.columns:
            noise_options = sorted(df["noise_applied"].unique().tolist())
            default_noise = [True] if True in noise_options else noise_options
            selected_noise = st.multiselect("Noise applied", options=noise_options, default=default_noise)
            df = df[df["noise_applied"].isin(selected_noise)]
    with col_f3:
        if "total_pulses" in df.columns:
            pulse_options = sorted(df["total_pulses"].unique().tolist())
            selected_pulses = st.multiselect("Total pulses", options=pulse_options, default=[max(pulse_options)])
            df = df[df["total_pulses"].isin(selected_pulses)]
    with col_f4:
        if "distance_km" in df.columns:
            dist_options = sorted(df["distance_km"].unique().tolist())
            selected_dist = st.multiselect("Distance (km)", options=dist_options, default=dist_options)
            df = df[df["distance_km"].isin(selected_dist)]

    # ── KPI summary (for current filter) ─────────────────────────────────
    if not df.empty and "secure_key_rate" in df.columns:
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        col_k1.metric("Rows", f"{len(df):,}")
        col_k2.metric("Mean SKR", f"{df['secure_key_rate'].mean():.4g}")
        col_k3.metric("Max SKR",  f"{df['secure_key_rate'].max():.4g}")
        col_k4.metric("Mean QBER", f"{df['qber'].mean():.4g}" if "qber" in df.columns else "—")

    # ── Data preview + CSV download ─────────────────────────────────────
    st.markdown("#### Data Preview")
    st.dataframe(df, use_container_width=True, height=320)
    st.download_button(
        "⬇️ Download filtered table (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="qkd_results_filtered.csv", mime="text/csv",
    )

    if df.empty:
        st.info("Filtered dataframe is empty — relax filters to see plots.")
        return

    # ── Build all academic plots ────────────────────────────────────────
    plots: list[tuple[str, go.Figure | None]] = [
        ("Secure Key Rate vs Distance",     fig_skr_vs_distance(df)),
        ("QBER vs Distance",                fig_qber_vs_distance(df)),
        ("Detection Yield vs Distance",     fig_yield_vs_distance(df)),
        ("Channel Loss Budget (DWDM)",       fig_loss_budget(df)),
        ("Finite-Size Contour",             fig_finite_size_contour(df)),
        ("Sweep Sensitivity",               fig_sweep_sensitivity(df)),
        ("Convergence Audit",               fig_convergence_audit(df)),
    ]
    plots = [(name, fig) for name, fig in plots if fig is not None]
    if not plots:
        st.info("No plottable data with the current selection.")
        return

    # ── Render plots with per-plot export ───────────────────────────────
    plot_choices = [name for name, _ in plots]
    selected_plot_names = st.multiselect(
        "Select plots to display", options=plot_choices, default=plot_choices,
    )
    for name, fig in plots:
        if name not in selected_plot_names:
            continue
        st.markdown(f"#### {name}")
        st.plotly_chart(fig, use_container_width=True)
        _render_plot_export_row(fig, name)


def _render_plot_export_row(fig: go.Figure, name: str) -> None:
    """Render a small download row for a Plotly figure: PNG + PDF."""
    col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 4])
    base_name = name.lower().replace(" ", "_").replace("/", "_")
    with col_dl1:
        try:
            png_bytes = export_plotly_image(fig, fmt="png")
            st.download_button(
                "⬇️ PNG (300 dpi)",
                data=png_bytes, mime="image/png",
                file_name=f"{base_name}.png",
                key=f"dl_png_{base_name}",
            )
        except Exception as exc:
            st.caption(f"PNG export unavailable: {exc}")
    with col_dl2:
        try:
            pdf_bytes = export_plotly_image(fig, fmt="pdf")
            st.download_button(
                "⬇️ PDF (vector)",
                data=pdf_bytes, mime="application/pdf",
                file_name=f"{base_name}.pdf",
                key=f"dl_pdf_{base_name}",
            )
        except Exception as exc:
            st.caption(f"PDF export unavailable: {exc}")


# ─────────────────────────────────────────────────────────────────────────
# Run-files browser
# ─────────────────────────────────────────────────────────────────────────
def render_files_panel() -> None:
    st.subheader("Run Files")
    run_dir_value = st.session_state.last_run_dir
    if not run_dir_value:
        st.info("No run has been completed yet.")
        return
    run_dir = Path(run_dir_value)
    if not run_dir.exists():
        st.warning("Run folder not found.")
        return
    files = sorted(
        [p for p in run_dir.glob("**/*") if p.is_file()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not files:
        st.info("No files in this run.")
        return
    for path in files:
        relative = path.relative_to(run_dir)
        size_kb = path.stat().st_size / 1024
        with st.expander(f"{relative} — {size_kb:.1f} KB"):
            if path.suffix.lower() in [".json", ".csv", ".log", ".txt"]:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    st.code(content[:20_000],
                            language="json" if path.suffix.lower() == ".json" else "text")
                except Exception as exc:
                    st.warning(f"Cannot preview file: {exc}")
            st.download_button(
                label=f"Download {relative.name}",
                data=path.read_bytes(),
                file_name=relative.name,
                key=f"download-{path}",
            )


# ─────────────────────────────────────────────────────────────────────────
# Help / Integration notes
# ─────────────────────────────────────────────────────────────────────────
def render_help_panel() -> None:
    st.subheader("Integration Notes")
    st.markdown(
        """
This dashboard calls ``main_optimized.run_and_save_csv()`` directly — no
subprocess or CLI bridge is needed. The simulation config dict and sweep dict
are passed as native Python objects. DWDM and auto-optimize flags are
forwarded as keyword arguments.

#### Key points

1. **Config**: The base configuration is imported from
   ``main_optimized.DEFAULT_CONFIG`` — the UI always reflects the engine's
   true defaults. Every parameter documented in SIMULATION_SPEC.md §2 is
   editable either directly on a Setup tab or via the Raw JSON editor.
2. **Sweep**: The sweep dict is passed directly to
   ``run_and_save_csv(sweeps=...)``. In Single Point mode the grid is
   collapsed to a single (distance, pulses) triple per combo.
3. **Output**: The CSV file is automatically discovered in the run
   directory by the Results Explorer tab. The default file name is
   ``qkd_results_optimized_sweep.csv``.
4. **Caching**: Single-point runs are wrapped in
   ``@st.cache_data`` keyed on the JSON-serialised config + flags, so
   flipping sliders back and forth is responsive.
5. **Validation**: Spec-mandated constraints (Σ ε < 1, P_signal +
   P_decoy + P_vacuum = 1, bias > breakdown, min ≤ max pulses, etc.) are
   enforced as inline warnings before run.
6. **Exports**: Filtered sweep table downloads as CSV; every plot
   downloads as PNG (300 dpi, raster) or PDF (vector) via the
   optional ``kaleido`` package.
        """
    )

    st.markdown("---")
    st.markdown("#### Plot catalogue (spec §4.2)")
    plot_table = [
        ("§4.2.1", "Secure Key Rate vs Distance", "log10 SKR; noise ON/OFF series; Rogers cross-check"),
        ("§4.2.2", "QBER vs Distance",             "log10 QBER; CI band; 11% threshold"),
        ("§4.2.3", "Detection Yield vs Distance",   "log10 yields for signal/decoy/vacuum"),
        ("§4.2.4", "Channel Loss Budget (DWDM)",    "Loss (dB) + Raman dark rate (Hz, twin axis)"),
        ("§4.2.5", "Finite-Size Contour",            "Heatmap of log10 SKR over (L, log10 N)"),
        ("§4.2.6", "Sweep Sensitivity",              "SKR vs swept param; argmax marker"),
        ("§4.2.7", "Convergence Audit",              "Per-row wall time; failure-rate bar"),
    ]
    st.table(pd.DataFrame(plot_table, columns=["Spec §", "Plot", "Notes"]))


# ─────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    init_page()
    ensure_state()
    render_header()

    (output_dir, workers, mode, dwdm, auto_optimize,
     wdm_channel_count, wdm_channel_power_dbm, wdm_raman_coefficient) = render_sidebar()

    tab_setup, tab_execute, tab_results, tab_files, tab_help = st.tabs([
        "⚙️ Setup",
        "🚀 Execute",
        "📊 Results",
        "📁 Files",
        "❓ Help",
    ])

    with tab_setup:
        current_config = render_config_editor()
        current_sweep = render_sweep_editor(mode)
        st.session_state.config = current_config
        st.session_state.sweep = current_sweep

    with tab_execute:
        config_to_run = st.session_state.get("config", json.loads(json.dumps(DEFAULT_CONFIG)))
        sweep_to_run = st.session_state.get("sweep", json.loads(json.dumps(DEFAULT_SWEEP)))
        render_execute_panel(
            config=config_to_run,
            sweep=sweep_to_run,
            output_dir=output_dir,
            workers=workers,
            mode=mode,
            dwdm=dwdm,
            auto_optimize=auto_optimize,
            wdm_channel_count=wdm_channel_count,
            wdm_channel_power_dbm=wdm_channel_power_dbm,
            wdm_raman_coefficient=wdm_raman_coefficient,
        )

    with tab_results:
        render_results_panel()

    with tab_files:
        render_files_panel()

    with tab_help:
        render_help_panel()


if __name__ == "__main__":
    main()
