from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import io

import concurrent.futures
import pandas as pd
import plotly.express as px
import streamlit as st


APP_TITLE = "QKD Simulation Dashboard"
DEFAULT_OUTPUT_DIR = "qkd_ui_runs"


from main_optimized import (
    DEFAULT_CONFIG,
    run_and_save_csv,
)
from qkd.proofs.optimization import (
    optimize_decoy_intensities,
    calculate_optimal_mu_ma2005,
    optimize_rotation_angle,
    optimize_rotation_angle_chapman,
)

# Master list of ALL available sweep parameters with their default values
MASTER_SWEEP_CATALOG: dict[str, dict[str, Any]] = {
    # Source pulses
    "source.pulses.signal.mu": {"default": [0.3, 0.5, 0.8], "group": "Source"},
    "source.pulses.decoy.mu": {"default": [0.05, 0.1, 0.2], "group": "Source"},
    # Detector efficiency
    "detector.det_eff_d0": {"default": [0.10, 0.15, 0.20], "group": "Detector"},
    "detector.det_eff_d1": {"default": [0.10, 0.15, 0.20], "group": "Detector"},
    # Detector noise
    "detector.dark_rate": {"default": [1e-7, 1e-6, 1e-5], "group": "Detector"},
    "detector.qber_intrinsic": {"default": [0.005, 0.01, 0.02], "group": "Detector"},
    "detector.misalignment": {"default": [0.002, 0.005, 0.01], "group": "Detector"},
    # Detector advanced
    "detector.bias_voltage": {"default": [44.0, 50.0], "group": "Detector"},
    "detector.temperature_k": {"default": [77.0, 293.0], "group": "Detector"},
    "detector.afterpulse_lifetime_ns": {"default": [10.0, 100.0, 500.0], "group": "Detector"},
    "detector.detector_type": {"default": ["SPD", "PNRD", "SNSPD"], "group": "Detector"},
    "detector.dead_time_model": {"default": ["NON_PARALYZABLE"], "group": "Detector"},
    "detector.afterpulse_model": {"default": ["EXPONENTIAL", "GEOMETRIC"], "group": "Detector"},
    "detector.double_click_policy": {"default": ["RANDOM", "DISCARD", "rogers_2007"], "group": "Detector"},
    "detector.strict_mode": {"default": [False, True], "group": "Detector"},
    # Source properties
    "source.intensity_jitter": {"default": [0.0, 0.05, 0.1, 0.15], "group": "Source"},
    "source.modulation_index": {"default": [0.90, 0.95, 0.99, 1.0], "group": "Source"},
    "source.statistics_type": {"default": ["POISSON", "THERMAL"], "group": "Source"},
    "source.error_model": {"default": ["RANDOM_GAUSSIAN", "ADVERSARIAL_BLOCK"], "group": "Source"},
    "source.intended_proof": {"default": ["LIM_2014", "MA_2005", "WANG_2005", "TIGHT"], "group": "Source"},
    "source.confidence_method": {"default": ["GAUSSIAN", "HOEFFDING", "CLOPPER_PEARSON"], "group": "Source"},
    "source.ideal_emission_probability": {"default": [0.95, 0.99, 1.0], "group": "Source"},
    "source.use_linear_modulation_approximation": {"default": [False, True], "group": "Source"},
    "source.N_channels": {"default": [1, 2, 4], "group": "Source"},
    "source.is_bidirectional": {"default": [False, True], "group": "Source"},
    "source.density_matrices": {
        "default": [None, {"signal": [[1.0, 0.0], [0.0, 0.0]], "decoy": [[1.0, 0.0], [0.0, 0.0]], "vacuum": [[1.0, 0.0], [0.0, 0.0]]}],
        "group": "Source"
    },
    # Channel
    "channel.dispersion_parameter_ps_nm_km": {"default": [0.0, 17.0], "group": "Channel"},
    # Protocol
    "protocol_runtime.protocol_class": {"default": ["BB84DecoyProtocol", "RedundantTransmissionProtocol"], "group": "Protocol"},
    "protocol_runtime.use_entangling_decoder": {"default": [False, True], "group": "Protocol"},
    "protocol_runtime.redundancy_M": {"default": [2, 3], "group": "Protocol"},
}

# Default active sweeps (subset for quick start)
DEFAULT_SWEEP: dict[str, list[Any]] = {
    "source.pulses.signal.mu": [0.3, 0.5, 0.8],
    "detector.det_eff_d0": [0.10, 0.15, 0.20],
    "detector.dark_rate": [1e-7, 1e-6, 1e-5],
    "source.intensity_jitter": [0.0, 0.05, 0.1, 0.15],
    "source.modulation_index": [0.90, 0.95, 0.99, 1.0],
}



@dataclass
class RunPaths:
    run_dir: Path
    config_path: Path
    sweep_path: Path
    stdout_path: Path
    stderr_path: Path


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
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.35rem;
        }
        .small-muted {
            color: #666;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_state() -> None:
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


def parse_scalar(raw: str) -> Any:
    value = raw.strip()

    if value == "":
        return ""

    lowered = value.lower()

    if lowered == "true":
        return True

    if lowered == "false":
        return False

    if lowered == "none" or lowered == "null":
        return None

    # Try JSON for complex types (dicts, nested lists)
    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    try:
        if any(marker in value for marker in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value

def parse_sweep_value_list(raw: str) -> list[Any]:
    text = raw.strip()

    if not text:
        return []

    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("Sweep value must be a list.")
        return parsed

    # Handle comma-separated values, preserving None/bool/complex
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(parse_scalar(part))
    return values





def build_sweep_from_rows(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
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
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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
                for key in ["results", "data", "records", "simulations"]:
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


def render_header() -> None:
    st.title("🔐 QKD Simulation Dashboard")
    st.caption(
        "A research dashboard for configuring parameters, running sweeps, viewing logs, and analyzing CSV/JSON outputs from the QKD simulation."
    )


def render_sidebar() -> tuple[str, int]:
    st.sidebar.header("Execution")

    output_dir = st.sidebar.text_input(
        "Output directory",
        value=DEFAULT_OUTPUT_DIR,
        help="Each run is stored in a separate folder within this path.",
    )

    workers = st.sidebar.number_input(
        "Workers",
        min_value=1,
        max_value=max(os.cpu_count() or 1, 1),
        value=max((os.cpu_count() or 2) - 1, 1),
        step=1,
        help=f"Number of parallel processes. Your system has {os.cpu_count()} CPUs.",
    )

    return output_dir, int(workers)


def render_config_editor() -> dict[str, Any]:
    st.subheader("Base Configuration")
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # Deep copy

    tab_proto, tab_channel, tab_detector, tab_source, tab_sim = st.tabs([
        "Protocol", "Channel", "Detector", "Source", "Simulation"
    ])

    # ==================== PROTOCOL TAB ====================
    with tab_proto:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            config["protocol_runtime"]["protocol_class"] = st.selectbox(
                "Protocol Class",
                options=["BB84DecoyProtocol", "B92Protocol", "MDIQKDProtocol", "RedundantTransmissionProtocol"],
                index=0,
            )
            config["protocol_params"]["error_correction_efficiency"] = st.number_input(
                "Error Correction Efficiency (f_ec)",
                min_value=1.0, max_value=3.0,
                value=config["protocol_params"]["error_correction_efficiency"],
                step=0.01,
                help="Cascade efficiency. Typical: 1.16",
            )
        with col_p2:
            config["protocol_runtime"]["parameter_estimation_fraction"] = st.number_input(
                "Parameter Estimation Fraction",
                min_value=0.01, max_value=0.5,
                value=config["protocol_runtime"]["parameter_estimation_fraction"],
                step=0.01,
            )
            config["protocol_runtime"]["num_worker_rngs"] = st.number_input(
                "Num Worker RNGs",
                min_value=1, max_value=16,
                value=config["protocol_runtime"]["num_worker_rngs"],
                step=1,
            )

        # Basis probabilities
        with st.expander("Basis Probabilities"):
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                config["protocol_runtime"]["alice_z_basis_prob"] = st.number_input(
                    "Alice Z-Basis Prob",
                    min_value=0.0, max_value=1.0,
                    value=config["protocol_runtime"]["alice_z_basis_prob"],
                    step=0.01,
                )
            with col_b2:
                config["protocol_runtime"]["bob_z_basis_prob"] = st.number_input(
                    "Bob Z-Basis Prob",
                    min_value=0.0, max_value=1.0,
                    value=config["protocol_runtime"]["bob_z_basis_prob"],
                    step=0.01,
                )
            with col_b3:
                config["protocol_runtime"]["z_basis_prob"] = st.number_input(
                    "Z-Basis Prob",
                    min_value=0.0, max_value=1.0,
                    value=config["protocol_runtime"]["z_basis_prob"],
                    step=0.01,
                )

        # Double click policies
        with st.expander("Double Click Policies"):
            col_dc1, col_dc2 = st.columns(2)
            with col_dc1:
                config["protocol_runtime"]["double_click_policy"] = st.selectbox(
                    "Protocol Double Click Policy",
                    options=["DISCARD", "RANDOM"],
                    index=0,
                )
            with col_dc2:
                config["detector"]["double_click_policy"] = st.selectbox(
                    "Detector Double Click Policy",
                    options=["RANDOM", "DISCARD", "rogers_2007"],
                    index=0,
                    help="rogers_2007 enables Rogers et al. (2007) basis-level paralyzable sifting (replaces deprecated PARALYZABLE dead_time_model).",
                )

        # Entangling decoder options (for RedundantTransmissionProtocol)
        with st.expander("Entangling Decoder (RedundantTransmissionProtocol)"):
            config["protocol_runtime"]["use_entangling_decoder"] = st.checkbox(
                "Use Entangling Decoder",
                value=config["protocol_runtime"]["use_entangling_decoder"],
            )
            config["protocol_runtime"]["use_entangling_encoder"] = st.checkbox(
                "Use Entangling Encoder",
                value=config["protocol_runtime"]["use_entangling_encoder"],
            )
            config["protocol_runtime"]["redundancy_M"] = st.number_input(
                "Redundancy M",
                min_value=1, max_value=10,
                value=config["protocol_runtime"]["redundancy_M"],
                step=1,
            )
            codeword_str = st.text_input(
                "Codeword Mapping (JSON or None)",
                value="",
                help="Leave empty for None, or provide JSON dict",
            )
            config["protocol_runtime"]["codeword_mapping"] = json.loads(codeword_str) if codeword_str.strip() else None
            
            damping_val = st.number_input(
                "Damping Parameter",
                min_value=0.0, max_value=1.0,
                value=config["protocol_runtime"].get("damping_parameter") or 0.0,
                step=0.01,
            )
            config["protocol_runtime"]["damping_parameter"] = damping_val if damping_val > 0 else None
            
            rot_str = st.text_input(
                "Rotation Angle (radians or None)",
                value="",
                help="Leave empty for None",
            )
            config["protocol_runtime"]["rotation_angle"] = float(rot_str) if rot_str.strip() else None

        # Security Epsilons
        with st.expander("Security Epsilons (ALL 6)"):
            eps = config["protocol"]["epsilons"]
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                eps["eps_sec"] = st.number_input("ε_sec", value=eps["eps_sec"], format="%.1e")
                eps["eps_cor"] = st.number_input("ε_cor", value=eps["eps_cor"], format="%.1e")
            with col_e2:
                eps["eps_pe"] = st.number_input("ε_pe", value=eps["eps_pe"], format="%.1e")
                eps["eps_smooth"] = st.number_input("ε_smooth", value=eps["eps_smooth"], format="%.1e")
            with col_e3:
                eps["eps_pa"] = st.number_input("ε_pa", value=eps["eps_pa"], format="%.1e")
                eps["eps_phase_est"] = st.number_input("ε_phase_est", value=eps["eps_phase_est"], format="%.1e")

    # ==================== CHANNEL TAB ====================
    with tab_channel:
        col_ch1, col_ch2 = st.columns(2)
        with col_ch1:
            config["channel"]["fiber_loss_db_km"] = st.number_input(
                "Fiber Loss (dB/km)",
                min_value=0.0, max_value=2.0,
                value=config["channel"]["fiber_loss_db_km"],
                step=0.01,
                help="Standard telecom fiber: 0.2 dB/km",
            )
        with col_ch2:
            config["channel"]["dispersion_parameter_ps_nm_km"] = st.number_input(
                "Chromatic Dispersion (ps/nm/km)",
                min_value=0.0, max_value=100.0,
                value=config["channel"]["dispersion_parameter_ps_nm_km"],
                step=0.1,
                help="Set to 0.0 to disable dispersion broadening. Standard SMF-28: 17.0",
            )

    # ==================== DETECTOR TAB ====================
    with tab_detector:
        # Basic efficiency and dark counts
        st.markdown("#### Basic Parameters")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            config["detector"]["det_eff_d0"] = st.number_input(
                "Detector 0 Efficiency",
                min_value=0.0, max_value=1.0,
                value=config["detector"]["det_eff_d0"],
                step=0.01,
            )
            config["detector"]["dark_rate"] = st.number_input(
                "Dark Count Rate D0 (per gate)",
                min_value=0.0, max_value=1000.0,
                value=config["detector"]["dark_rate"],
                step=1e-7,
                format="%.2e",
                help="PROBABILITY per detection window",
            )
        with col_d2:
            config["detector"]["det_eff_d1"] = st.number_input(
                "Detector 1 Efficiency",
                min_value=0.0, max_value=1000.0,
                value=config["detector"]["det_eff_d1"],
                step=0.01,
            )
            dark_d1_val = st.number_input(
                "Dark Count Rate D1 (per gate)",
                min_value=0.0, max_value=5000.0,
                value=config["detector"]["dark_rate_d1"] or 0.0,
                step=1e-7,
                format="%.2e",
                help="Set to 0 to use same as D0",
            )
            config["detector"]["dark_rate_d1"] = dark_d1_val if dark_d1_val > 0 else None
        with col_d3:
            config["detector"]["qber_intrinsic"] = st.number_input(
                "Intrinsic QBER",
                min_value=0.0, max_value=0.1,
                value=config["detector"]["qber_intrinsic"],
                step=0.001,
            )
            config["detector"]["misalignment"] = st.number_input(
                "Misalignment Error",
                min_value=0.0, max_value=0.1,
                value=config["detector"]["misalignment"],
                step=0.001,
                format="%.4f",
            )

        # Detector type and models
        st.markdown("#### Detector Type & Models")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            config["detector"]["detector_type"] = st.selectbox(
                "Detector Type",
                options=["SPD", "PNRD", "SNSPD"],
                index=0,
            )
            config["detector"]["dead_time_model"] = st.selectbox(
                "Dead Time Model",
                options=["NON_PARALYZABLE", "PARALYZABLE"],
                index=0,
            )
        with col_m2:
            config["detector"]["afterpulse_model"] = st.selectbox(
                "Afterpulse Model",
                options=["EXPONENTIAL", "GEOMETRIC"],
                index=0,
            )
            config["detector"]["strict_mode"] = st.checkbox(
                "Strict Mode",
                value=config["detector"]["strict_mode"],
                help="Enable strict validation",
            )
        with col_m3:
            config["protocol_params"]["detector_type"] = config["detector"]["detector_type"]

        # Timing parameters
        st.markdown("#### Timing Parameters")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            config["detector"]["dead_time_ns"] = st.number_input(
                "Dead Time (ns)",
                min_value=0.0, max_value=1000.0,
                value=config["detector"]["dead_time_ns"],
                step=1.0,
                help="Set to 0 to disable dead time",
            )
        with col_t2:
            config["detector"]["jitter_fwhm_ns"] = st.number_input(
                "Jitter FWHM (ns)",
                min_value=0.0, max_value=1000.0,
                value=config["detector"]["jitter_fwhm_ns"],
                step=1.0,
                help="Set to 0 to disable jitter",
            )
        with col_t3:
            config["detector"]["afterpulse_prob"] = st.number_input(
                "Afterpulse Probability",
                min_value=0.0, max_value=0.1,
                value=config["detector"]["afterpulse_prob"],
                step=0.001,
                format="%.4f",
            )
            config["detector"]["afterpulse_lifetime_ns"] = st.number_input(
                "Afterpulse Lifetime (ns)",
                min_value=1.0, max_value=10000.0,
                value=config["detector"]["afterpulse_lifetime_ns"],
                step=10.0,
            )

        # Temperature and bias voltage (affects dark count scaling)
        with st.expander("Temperature & Bias Voltage (Affects Dark Count Scaling)"):
            st.caption("These parameters affect dynamic dark count rate calculation. Setting bias_voltage ≤ breakdown_voltage returns 0.0 dark rate.")
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                config["detector"]["temperature_k"] = st.number_input(
                    "Operating Temperature (K)",
                    min_value=0.0, max_value=400.0,
                    value=config["detector"]["temperature_k"],
                    step=1.0,
                )
                config["detector"]["ref_temperature_k"] = st.number_input(
                    "Reference Temperature (K)",
                    min_value=0.0, max_value=400.0,
                    value=config["detector"]["ref_temperature_k"],
                    step=1.0,
                )
            with col_v2:
                config["detector"]["bias_voltage"] = st.number_input(
                    "Bias Voltage (V)",
                    min_value=0.0, max_value=100.0,
                    value=config["detector"]["bias_voltage"],
                    step=0.5,
                    help="⚠️ If ≤ breakdown_voltage, dark rate becomes 0.0",
                )
                config["detector"]["breakdown_voltage"] = st.number_input(
                    "Breakdown Voltage (V)",
                    min_value=0.0, max_value=100.0,
                    value=config["detector"]["breakdown_voltage"],
                    step=0.5,
                )
                config["detector"]["ref_bias_voltage"] = st.number_input(
                    "Reference Bias Voltage (V)",
                    min_value=0.0, max_value=100.0,
                    value=config["detector"]["ref_bias_voltage"],
                    step=0.5,
                )

    # ==================== SOURCE TAB ====================
    with tab_source:
        # Pulse intensities
        st.markdown("#### Pulse Intensities")
        col_s, col_d, col_v = st.columns(3)
        with col_s:
            config["source"]["pulses"]["signal"]["mu"] = st.number_input(
                "Signal μ",
                min_value=0.0, max_value=2.0,
                value=config["source"]["pulses"]["signal"]["mu"],
                step=0.01,
            )
            config["source"]["pulses"]["signal"]["prob"] = st.number_input(
                "Signal Prob",
                min_value=0.0, max_value=1.0,
                value=config["source"]["pulses"]["signal"]["prob"],
                step=0.01,
            )
        with col_d:
            config["source"]["pulses"]["decoy"]["mu"] = st.number_input(
                "Decoy μ",
                min_value=0.0, max_value=2.0,
                value=config["source"]["pulses"]["decoy"]["mu"],
                step=0.01,
            )
            config["source"]["pulses"]["decoy"]["prob"] = st.number_input(
                "Decoy Prob",
                min_value=0.0, max_value=1.0,
                value=config["source"]["pulses"]["decoy"]["prob"],
                step=0.01,
            )
        with col_v:
            config["source"]["pulses"]["vacuum"]["mu"] = st.number_input(
                "Vacuum μ",
                min_value=0.0, max_value=0.01,
                value=config["source"]["pulses"]["vacuum"]["mu"],
                step=0.001,
                format="%.4f",
            )
            config["source"]["pulses"]["vacuum"]["prob"] = st.number_input(
                "Vacuum Prob",
                min_value=0.0, max_value=1.0,
                value=config["source"]["pulses"]["vacuum"]["prob"],
                step=0.01,
            )

        total_prob = sum(p["prob"] for p in config["source"]["pulses"].values())
        if abs(total_prob - 1.0) > 0.01:
            st.error(f"⚠️ Pulse probabilities sum to {total_prob:.3f}, should be 1.0!")

        # Source basic properties
        st.markdown("#### Source Properties")
        col_sp1, col_sp2, col_sp3 = st.columns(3)
        with col_sp1:
            config["source"]["pulse_period_ns"] = st.number_input(
                "Pulse Period (ns)",
                min_value=0.1, max_value=1000.0,
                value=config["source"]["pulse_period_ns"],
                step=0.1,
                help="1 ns = 1 GHz repetition rate",
            )
            config["source"]["intensity_jitter"] = st.number_input(
                "Intensity Jitter",
                min_value=0.0, max_value=1.0,
                value=config["source"]["intensity_jitter"],
                step=0.01,
            )
        with col_sp2:
            config["source"]["modulation_index"] = st.number_input(
                "Modulation Index",
                min_value=0.0, max_value=1.0,
                value=config["source"]["modulation_index"],
                step=0.01,
            )
            config["source"]["extinction_ratio"] = st.number_input(
                "Extinction Ratio (dB)",
                min_value=0.0, max_value=100.0,
                value=config["source"]["extinction_ratio"],
                step=1.0,
            )
        with col_sp3:
            config["source"]["ideal_emission_probability"] = st.number_input(
                "Ideal Emission Probability",
                min_value=0.0, max_value=1.0,
                value=config["source"]["ideal_emission_probability"],
                step=0.01,
            )
            config["source"]["source_fidelity"] = st.number_input(
                "Source Fidelity",
                min_value=0.0, max_value=1.0,
                value=config["source"]["source_fidelity"],
                step=0.001,
                format="%.4f",
            )

        # Statistics and error model
        st.markdown("#### Statistics & Error Model")
        col_se1, col_se2, col_se3 = st.columns(3)
        with col_se1:
            config["source"]["statistics_type"] = st.selectbox(
                "Statistics Type",
                options=["POISSON", "THERMAL"],
                index=0,
            )
        with col_se2:
            config["source"]["error_model"] = st.selectbox(
                "Error Model",
                options=["RANDOM_GAUSSIAN", "ADVERSARIAL_BLOCK"],
                index=0,
            )
        with col_se3:
            config["source"]["adversarial_block_size"] = st.number_input(
                "Adversarial Block Size",
                min_value=1, max_value=100000,
                value=config["source"]["adversarial_block_size"],
                step=100,
            )

        # Channel configuration
        with st.expander("Channel Configuration"):
            col_cc1, col_cc2 = st.columns(2)
            with col_cc1:
                config["source"]["N_channels"] = st.number_input(
                    "N Channels",
                    min_value=1, max_value=100,
                    value=config["source"]["N_channels"],
                    step=1,
                )
                config["source"]["is_bidirectional"] = st.checkbox(
                    "Is Bidirectional",
                    value=config["source"]["is_bidirectional"],
                )
            with col_cc2:
                config["source"]["preferred_lp_solver"] = st.selectbox(
                    "LP Solver",
                    options=["highs", "scipy", "cvxopt"],
                    index=0,
                )
                config["source"]["expected_decoder"] = st.selectbox(
                    "Expected Decoder",
                    options=["LOCAL", "ENTANGLING"],
                    index=0,
                )
                config["source"]["assumed_double_click"] = st.selectbox(
                    "Assumed Double Click",
                    options=["RANDOM", "DISCARD"],
                    index=0,
                )

        # Approximation toggles
        with st.expander("Approximation Toggles"):
            config["source"]["use_small_angle_approximation"] = st.checkbox(
                "Use Small Angle Approximation",
                value=config["source"]["use_small_angle_approximation"],
            )
            config["source"]["use_linear_modulation_approximation"] = st.checkbox(
                "Use Linear Modulation Approximation",
                value=config["source"]["use_linear_modulation_approximation"],
            )

        # MZM Config
        with st.expander("MZM Modulator"):
            st.caption("Mach-Zehnder Modulator modeling for intensity modulation.")
            mzm = config["source"]["mzm"]
            col_mzm1, col_mzm2 = st.columns(2)
            with col_mzm1:
                mzm["extinction_ratio_db"] = st.number_input(
                    "MZM Extinction Ratio (dB)",
                    min_value=0.0, max_value=100.0,
                    value=mzm["extinction_ratio_db"],
                    step=1.0,
                )
                mzm["v_pi"] = st.number_input(
                    "MZM V_pi (V)",
                    min_value=0.1, max_value=20.0,
                    value=mzm["v_pi"],
                    step=0.1,
                )
            with col_mzm2:
                mzm["bias_voltage"] = st.number_input(
                    "MZM Bias Voltage (V)",
                    min_value=0.0, max_value=20.0,
                    value=mzm["bias_voltage"],
                    step=0.1,
                )
                mzm["rf_amplitude_v"] = st.number_input(
                    "MZM RF Amplitude (V)",
                    min_value=0.0, max_value=20.0,
                    value=mzm["rf_amplitude_v"],
                    step=0.1,
                )

        # Electrical Noise Config
        with st.expander("Electrical Noise (Driver)"):
            st.caption("Driver voltage noise modeling.")
            en = config["source"]["electrical_noise"]
            col_en1, col_en2 = st.columns(2)
            with col_en1:
                en["voltage_std_v"] = st.number_input(
                    "Voltage Std (V)",
                    min_value=0.0, max_value=1.0,
                    value=en["voltage_std_v"],
                    step=0.01,
                    format="%.4f",
                )
            with col_en2:
                en["bandwidth_hz"] = st.number_input(
                    "Noise Bandwidth (Hz)",
                    min_value=1e3, max_value=1e12,
                    value=en["bandwidth_hz"],
                    step=1e8,
                    format="%.2e",
                )

        # Source physical parameters (temperature, resistance, etc.)
        with st.expander("Source Physical Parameters"):
            col_pp1, col_pp2, col_pp3 = st.columns(3)
            with col_pp1:
                config["source"]["temperature_k"] = st.number_input(
                    "Source Temperature (K)",
                    min_value=0.0, max_value=500.0,
                    value=config["source"]["temperature_k"],
                    step=1.0,
                )
                config["source"]["v_pi"] = st.number_input(
                    "Source V_pi (V)",
                    min_value=0.1, max_value=20.0,
                    value=config["source"]["v_pi"],
                    step=0.1,
                )
            with col_pp2:
                config["source"]["bandwidth_hz"] = st.number_input(
                    "Source Bandwidth (Hz)",
                    min_value=1e3, max_value=1e12,
                    value=config["source"]["bandwidth_hz"],
                    step=1e8,
                    format="%.2e",
                )
                config["source"]["phi_bias"] = st.number_input(
                    "Phi Bias (rad)",
                    min_value=0.0, max_value=6.28,
                    value=config["source"]["phi_bias"],
                    step=0.01,
                    format="%.4f",
                )
            with col_pp3:
                config["source"]["driver_load_resistance_ohm"] = st.number_input(
                    "Driver Load Resistance (Ω)",
                    min_value=1.0, max_value=10000.0,
                    value=config["source"]["driver_load_resistance_ohm"],
                    step=10.0,
                )
                config["source"]["dc_photocurrent_a"] = st.number_input(
                    "DC Photocurrent (A)",
                    min_value=0.0, max_value=1e-3,
                    value=config["source"]["dc_photocurrent_a"],
                    step=1e-7,
                    format="%.2e",
                )
                config["source"]["max_intensity_mu"] = st.number_input(
                    "Max Intensity μ",
                    min_value=0.1, max_value=10.0,
                    value=config["source"]["max_intensity_mu"],
                    step=0.1,
                )

        # Security proof settings
        with st.expander("Security Proof Settings"):
            col_sec1, col_sec2 = st.columns(2)
            with col_sec1:
                config["source"]["intended_proof"] = st.selectbox(
                    "Intended Proof",
                    options=["LIM_2014", "MA_2005", "WANG_2005", "TIGHT"],
                    index=0,
                )
            with col_sec2:
                config["source"]["confidence_method"] = st.selectbox(
                    "Confidence Bound Method",
                    options=["GAUSSIAN", "CLOPPER_PEARSON", "HOEFFDING"],
                    index=0,
                )

        # Density Matrices (Advanced)
        with st.expander("Density Matrices (Advanced)"):
            st.caption("Provide custom density matrices for each pulse type. Leave empty to disable.")
            dm_str = st.text_area(
                "Density Matrices (JSON or empty)",
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

        # Security Metadata (Advanced)
        with st.expander("Security Metadata (Advanced)"):
            st.caption("Provide security metadata for the source. Leave empty to disable.")
            sm_str = st.text_area(
                "Security Metadata (JSON or empty)",
                value="",
                height=100,
            )
            if sm_str.strip():
                try:
                    config["source"]["security_metadata"] = json.loads(sm_str)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON for security metadata: {e}")
                    config["source"]["security_metadata"] = None
            else:
                config["source"]["security_metadata"] = None

    # ==================== SIMULATION TAB ====================
    with tab_sim:
        st.warning("⚠️ main_optimized runs MULTIPLE pulse counts (min_log to max_log) and distances!")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            config["simulation"]["min_pulses_log"] = st.number_input(
                "Min Pulses (log10)",
                min_value=4, max_value=12,
                value=config["simulation"]["min_pulses_log"],
                step=1,
                help="10^4 = 10,000 pulses",
            )
            config["simulation"]["max_pulses_log"] = st.number_input(
                "Max Pulses (log10)",
                min_value=4, max_value=12,
                value=config["simulation"]["max_pulses_log"],
                step=1,
                help="10^6 = 1,000,000 pulses",
            )
        with col_s2:
            config["simulation"]["distance_start_km"] = st.number_input(
                "Distance Start (km)",
                min_value=0.0, max_value=500.0,
                value=config["simulation"]["distance_start_km"],
            )
            config["simulation"]["distance_stop_km"] = st.number_input(
                "Distance Stop (km)",
                min_value=0.0, max_value=500.0,
                value=config["simulation"]["distance_stop_km"],
            )
            config["simulation"]["distance_points"] = st.number_input(
                "Distance Points",
                min_value=1, max_value=1000,
                value=config["simulation"]["distance_points"],
            )

        col_sim1, col_sim2, col_sim3 = st.columns(3)
        with col_sim1:
            config["simulation"]["rng_seed"] = st.number_input(
                "Random Seed",
                value=config["simulation"]["rng_seed"],
            )
        with col_sim2:
            config["simulation"]["apply_statistical_noise"] = st.checkbox(
                "Apply Statistical Noise",
                value=config["simulation"]["apply_statistical_noise"],
            )
        with col_sim3:
            config["simulation"]["sampling_cap_per_pulse_type"] = st.number_input(
                "Sampling Cap Per Pulse Type",
                min_value=1000, max_value=10000000,
                value=config["simulation"]["sampling_cap_per_pulse_type"],
                step=10000,
            )

        n_pulse_counts = config["simulation"]["max_pulses_log"] - config["simulation"]["min_pulses_log"] + 1
        n_distances = config["simulation"]["distance_points"]
        noise_multiplier = 2 if config["simulation"]["apply_statistical_noise"] else 1
        st.info(f"Per sweep combo: {n_pulse_counts} pulse counts × {noise_multiplier} noise settings × {n_distances} distances = **{n_pulse_counts * noise_multiplier * n_distances} simulations**")

        # ── Parameter Optimization Toggle ──
        with st.expander("🔧 Parameter Optimization (Ma 2005 / Chapman 2018)"):
            config["simulation"]["enable_optimization"] = st.checkbox(
                "Enable Parameter Optimization",
                value=config["simulation"].get("enable_optimization", False),
                help=(
                    "When enabled, optimize signal/decoy intensities (μ, ν) and rotation angles "
                    "before running the simulation. Uses Ma et al. 2005 for decoy intensities "
                    "and Chapman et al. 2018 for rotation angles. Optimized values are applied "
                    "to the config at the midpoint distance of the sweep range."
                ),
            )
            if config["simulation"]["enable_optimization"]:
                st.caption(
                    "Optimization will be applied at the **midpoint distance** of the configured range. "
                    "The optimized μ and ν values replace the current signal/decoy intensities. "
                    "For RedundantTransmissionProtocol, the rotation angle is also optimized."
                )
                proof = config["source"].get("intended_proof", "LIM_2014")
                if proof == "MA_2005":
                    st.info("→ Will use `calculate_optimal_mu_ma2005` + `optimize_decoy_intensities`")
                else:
                    st.info(f"→ Will use `optimize_decoy_intensities` (proof: {proof})")
                if config["protocol_runtime"]["protocol_class"] == "RedundantTransmissionProtocol":
                    st.info("→ Will also use `optimize_rotation_angle_chapman` for rotation angle")

    # Raw JSON Editor (always at bottom)
    with st.expander("Raw JSON Editor (Override All)"):
        raw = st.text_area("Edit JSON", value=json.dumps(config, indent=2), height=400)
        try:
            config = json.loads(raw)
            st.success("Valid JSON")
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")

    return config


def render_sweep_editor() -> dict[str, list[Any]]:
    st.subheader("Parameter Sweep")

    st.caption(
        "Use dotted paths for nested parameters, e.g., channel.fiber_length_km or detector.efficiency."
    )

    # Add new sweep parameter from catalog
    with st.expander("➕ Add Parameter from Catalog", expanded=False):
        # Group by category
        groups: dict[str, list[str]] = {}
        for param, info in MASTER_SWEEP_CATALOG.items():
            group = info["group"]
            groups.setdefault(group, []).append(param)

        col_add1, col_add2 = st.columns([2, 1])
        with col_add1:
            selected_group = st.selectbox("Category", options=list(groups.keys()))
            selected_param = st.selectbox(
                "Parameter",
                options=groups[selected_group],
                format_func=lambda x: x.replace(".", " → "),
            )
        with col_add2:
            # Show default values and let user edit
            default_vals = MASTER_SWEEP_CATALOG[selected_param]["default"]
            default_str = ", ".join(
                str(v) if not isinstance(v, (dict, list)) else json.dumps(v)
                for v in default_vals
            )
            new_values_str = st.text_area(
                "Values",
                value=default_str,
                height=80,
                help="Comma-separated or JSON list",
            )

        if st.button("Add to Sweep", type="primary"):
            # Check if already exists
            existing_params = [row.get("parameter", "") for row in st.session_state.sweep_rows]
            if selected_param in existing_params:
                st.warning(f"Parameter '{selected_param}' already exists in sweep table.")
            else:
                st.session_state.sweep_rows.append({
                    "enabled": True,
                    "parameter": selected_param,
                    "values": new_values_str,
                })
                st.rerun()

    # Add custom parameter
    with st.expander("✏️ Add Custom Parameter", expanded=False):
        col_cust1, col_cust2 = st.columns([1, 1])
        with col_cust1:
            custom_param = st.text_input(
                "Custom dotted path",
                placeholder="e.g., detector.custom_field",
            )
        with col_cust2:
            custom_values = st.text_input(
                "Values",
                placeholder="0.1, 0.2, 0.3",
            )
        if st.button("Add Custom"):
            if custom_param and custom_values:
                st.session_state.sweep_rows.append({
                    "enabled": True,
                    "parameter": custom_param,
                    "values": custom_values,
                })
                st.rerun()

    # Quick load buttons
    st.markdown("---")
    col_quick1, col_quick2, col_quick3 = st.columns(3)
    with col_quick1:
        if st.button("Load All Defaults", help="Load all 27 sweep parameters with defaults"):
            st.session_state.sweep_rows = [
                {
                    "enabled": True,
                    "parameter": param,
                    "values": ", ".join(
                        str(v) if not isinstance(v, (dict, list)) else json.dumps(v)
                        for v in info["default"]
                    ),
                }
                for param, info in MASTER_SWEEP_CATALOG.items()
            ]
            st.rerun()
    with col_quick2:
        if st.button("Clear All"):
            st.session_state.sweep_rows = []
            st.rerun()
    with col_quick3:
        if st.button("Reset to Quick Start"):
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
            st.rerun()

    st.markdown("---")

    # Editable table
    edited_rows = st.data_editor(
        st.session_state.sweep_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "enabled": st.column_config.CheckboxColumn("Enabled", default=True),
            "parameter": st.column_config.TextColumn(
                "Parameter",
                help="Example: channel.fiber_length_km",
            ),
            "values": st.column_config.TextColumn(
                "Values",
                help="Example: 0, 25, 50 or [0, 25, 50]",
            ),
        },
    )

    st.session_state.sweep_rows = list(edited_rows)

    try:
        sweep = build_sweep_from_rows(st.session_state.sweep_rows)
        combinations = count_combinations(sweep)

        metric_a, metric_b = st.columns(2)

        with metric_a:
            st.metric("Sweep parameters", len(sweep))

        with metric_b:
            st.metric("Total combinations", f"{combinations:,}")

        if combinations > 10_000:
            st.warning("The number of combinations is large and the simulation may take a long time to run.")

        if combinations > 100_000:
            st.error("🚨 More than 100,000 combinations! This may take hours to complete.")

        with st.expander("Sweep JSON preview"):
            st.json(sweep)

        # Show which catalog parameters are NOT in current sweep
        active_params = set(sweep.keys())
        missing_params = set(MASTER_SWEEP_CATALOG.keys()) - active_params
        if missing_params:
            with st.expander(f"Available but not active ({len(missing_params)} parameters)"):
                for group in ["Source", "Detector", "Channel", "Protocol"]:
                    group_params = [p for p in missing_params if MASTER_SWEEP_CATALOG[p]["group"] == group]
                    if group_params:
                        st.markdown(f"**{group}:**")
                        for p in group_params:
                            st.text(f"  • {p}")

        return sweep

    except Exception as exc:
        st.error(f"Sweep parsing error: {exc}")
        return {}



def _apply_optimization_to_config(config: dict[str, Any]) -> dict[str, Any]:
    """Apply parameter optimization to the config before simulation.

    Uses the midpoint distance of the configured range as the representative
    distance.  Dispatches to the appropriate optimization function based on
    the intended proof and protocol class.
    """
    sim = config["simulation"]
    dist_start = float(sim.get("distance_start_km", 0.0))
    dist_stop = float(sim.get("distance_stop_km", 150.0))
    rep_distance = (dist_start + dist_stop) / 2.0 if dist_stop > dist_start else dist_start

    fiber_loss = float(config["channel"]["fiber_loss_db_km"])
    det_eff = float(config["detector"]["det_eff_d0"])
    dark_rate = float(config["detector"]["dark_rate"])
    f_ec = float(config["protocol_params"]["error_correction_efficiency"])
    proof = config["source"].get("intended_proof", "LIM_2014")

    # ── Intensity optimization (all proofs) ──
    if proof == "MA_2005":
        e_detector = float(config["detector"].get("qber_intrinsic", 0.01)) + float(
            config["detector"].get("misalignment", 0.005)
        )
        mu_opt = calculate_optimal_mu_ma2005(
            distance_km=rep_distance,
            fiber_loss_db_km=fiber_loss,
            det_eff=det_eff,
            dark_rate=dark_rate,
            error_correction_f=f_ec,
            e_detector=e_detector,
        )
        # Also optimize decoy intensity
        _, nu_opt = optimize_decoy_intensities(
            distance_km=rep_distance,
            fiber_loss_db_km=fiber_loss,
            det_eff=det_eff,
            dark_rate=dark_rate,
            error_correction_f=f_ec,
        )
        config["source"]["pulses"]["signal"]["mu"] = mu_opt
        config["source"]["pulses"]["decoy"]["mu"] = nu_opt
    else:
        # LIM_2014, WANG_2005, TIGHT — use heuristic decoy optimization
        mu_opt, nu_opt = optimize_decoy_intensities(
            distance_km=rep_distance,
            fiber_loss_db_km=fiber_loss,
            det_eff=det_eff,
            dark_rate=dark_rate,
            error_correction_f=f_ec,
        )
        config["source"]["pulses"]["signal"]["mu"] = mu_opt
        config["source"]["pulses"]["decoy"]["mu"] = nu_opt

    # ── Rotation angle optimization (RedundantTransmissionProtocol only) ──
    if config["protocol_runtime"]["protocol_class"] == "RedundantTransmissionProtocol":
        # Gamma is the channel damping parameter (use damping_parameter if set, else 0)
        gamma = float(config["protocol_runtime"].get("damping_parameter") or 0.0)
        M = int(config["protocol_runtime"].get("redundancy_M", 2))
        theta_opt = optimize_rotation_angle_chapman(gamma=gamma, M=M)
        config["protocol_runtime"]["rotation_angle"] = theta_opt

    return config


def _run_simulation_in_dir(
    config: dict[str, Any],
    workers: int,
    sweep: dict[str, list[Any]],
    run_dir: Path,
    stdout_collector: io.StringIO,
) -> None:
    # ── Apply optimization if enabled ──
    if config.get("simulation", {}).get("enable_optimization", False):
        config = _apply_optimization_to_config(config)

    prev_cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        with contextlib.redirect_stdout(stdout_collector):
            run_and_save_csv(config=config, num_workers=workers, sweeps=sweep)
    finally:
        os.chdir(prev_cwd)

        
def render_run_panel(
    config: dict[str, Any],
    sweep: dict[str, list[Any]],
    output_dir: str,
    workers: int,
) -> None:
    st.subheader("Run Simulation")

    combinations = count_combinations(sweep) if sweep else 1

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Runs", combinations)
    with col_b:
        st.metric("Workers", workers)
    with col_c:
        st.metric("Output", output_dir)

    run_clicked = st.button(
        "🚀 Run simulation",
        type="primary",
        use_container_width=True,
    )

    if not run_clicked:
        return

    paths = create_run_paths(output_dir)
    write_json(paths.config_path, config)
    write_json(paths.sweep_path, sweep)

    st.session_state.last_run_dir = str(paths.run_dir)
    st.session_state.last_returncode = None

    st.info(f"Run directory: `{paths.run_dir}`")

    progress = st.progress(0)
    status_box = st.empty()
    log_box = st.empty()
    start_time = time.time()

    captured_lines: list[str] = []

    class _LineCollector(io.StringIO):
        """StringIO subclass that records each line for live display."""
        def write(self, text: str) -> int:
            if text.strip():
                captured_lines.append(text.rstrip())
            return super().write(text)

    collector = _LineCollector()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _run_simulation_in_dir,
            config=config,
            workers=workers,
            sweep=sweep,
            run_dir=paths.run_dir,
            stdout_collector=collector,
        )

        while not future.done():
            elapsed = time.time() - start_time
            progress.progress(min(0.95, elapsed / max(elapsed + 30.0, 1.0)))
            status_box.info(f"Running... elapsed: {elapsed:.1f}s")
            if captured_lines:
                log_box.code("\n".join(captured_lines[-80:]), language="text")
            time.sleep(0.5)

    # Simulation finished — check result
    try:
        future.result()
        progress.progress(1.0)
        st.session_state.last_returncode = 0
        status_box.success("Simulation finished successfully.")
    except Exception as exc:
        progress.progress(1.0)
        st.session_state.last_returncode = 1
        status_box.error(f"Simulation failed: {exc}")

    # Write captured output to log file
    try:
        paths.stdout_path.write_text(collector.getvalue(), encoding="utf-8")
    except Exception:
        pass

    # Confirm CSV location
    csv_name = "qkd_results_optimized_sweep.csv"
    csv_src = paths.run_dir / csv_name
    if csv_src.exists():
        st.info(f"Results: `{csv_src}`")

    elapsed = time.time() - start_time
    st.info(f"Elapsed: {elapsed:.1f}s")


def render_results_panel() -> None:
    st.subheader("Results Explorer")

    selected_run_dir = st.text_input(
        "Run directory",
        value=st.session_state.last_run_dir or "",
        help="Enter the path to the run output directory.",
    )

    if not selected_run_dir:
        st.info("No output directory selected.")
        return

    run_dir = Path(selected_run_dir).expanduser()

    if not run_dir.exists():
        st.error("Run directory does not exist.")
        return

    result_files = find_result_files(run_dir)

    if not result_files:
        st.warning("No CSV/JSON result files found in this folder.")
        return

    selected_file = st.selectbox(
        "Result file",
        options=result_files,
        format_func=lambda path: str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path),
    )

    df = load_result_dataframe(selected_file)

    if df is None or df.empty:
        st.warning("The selected file does not contain displayable data.")
        return

    df = normalize_dataframe_columns(df)

    # --- QKD SPECIFIC FILTERS ---
    st.markdown("#### QKD Data Filters")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        # Filter out ZERO_KEY status if the column exists
        if "status" in df.columns:
            valid_statuses = sorted(df["status"].unique().tolist())
            selected_status = st.multiselect(
                "Status (Hide ZERO_KEY for clean SKR plots)", 
                options=valid_statuses, 
                default=valid_statuses
            )
            df = df[df["status"].isin(selected_status)]

    with col_f2:
        if "noise_applied" in df.columns:
            noise_options = sorted(df["noise_applied"].unique().tolist())
            selected_noise = st.multiselect(
                "Noise Applied", 
                options=noise_options, 
                default=[True] if True in noise_options else noise_options
            )
            df = df[df["noise_applied"].isin(selected_noise)]

    with col_f3:
        if "total_pulses" in df.columns:
            pulse_options = sorted(df["total_pulses"].unique().tolist())
            selected_pulses = st.multiselect(
                "Total Pulses (Hide low stats like 10k)", 
                options=pulse_options, 
                default=[max(pulse_options)] # Default to highest pulse count
            )
            df = df[df["total_pulses"].isin(selected_pulses)]

    st.markdown("#### Data Preview")
    st.dataframe(df, use_container_width=True, height=320)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download FILTERED table as CSV",
        data=csv_data,
        file_name="qkd_results_filtered.csv",
        mime="text/csv",
    )

    numeric_columns = list(df.select_dtypes(include="number").columns)
    all_columns = list(df.columns)

    if not numeric_columns:
        st.info("Could not find a numeric column for the chart.")
        return

    # --- SMART COLUMN GUESSING ---
    distance_guess = guess_column(all_columns, ["fiber_length_km", "distance", "distance_km", "channel.fiber_length_km"])
    key_guess = guess_column(all_columns, ["secure_key_bits", "secure_key_rate", "secret_key_rate", "key_rate", "skr"])
    qber_guess = guess_column(all_columns, ["qber", "quantum_bit_error_rate", "error_rate"])

    # Automatically find sweep columns (e.g., sweep_detector_efficiency)
    sweep_columns = [col for col in all_columns if col.startswith("sweep_")]

    st.markdown("#### Interactive Plot")

    col_x, col_y, col_color = st.columns(3)

    with col_x:
        x_axis = st.selectbox(
            "X axis (Distance)", options=all_columns,
            index=all_columns.index(distance_guess) if distance_guess in all_columns else 0
        )

    with col_y:
        y_default = key_guess if key_guess in numeric_columns else numeric_columns[0]
        y_axis = st.selectbox(
            "Y axis (Metric)", options=numeric_columns,
            index=numeric_columns.index(y_default) if y_default in numeric_columns else 0
        )

    with col_color:
        # Prioritize sweep columns for coloring so we don't mix hardware
        color_options = ["None"] + sweep_columns + [col for col in all_columns if col not in sweep_columns]
        default_color_idx = 1 if sweep_columns else 0
        color_axis = st.selectbox("Color/group (Use sweep params!)", options=color_options, index=default_color_idx)

    # ========================================================
    # FIX: ISOLATE OTHER SWEEP PARAMETERS TO PREVENT ZIG-ZAGS
    # ========================================================
    if color_axis != "None" and color_axis in sweep_columns:
        # Find sweep columns that are NOT being used for coloring
        other_sweeps = [col for col in sweep_columns if col != color_axis]
        
        if other_sweeps:
            st.markdown("#### 🔒 Isolate Other Sweep Parameters")
            st.caption("To draw clean lines, lock the other sweep parameters to a single value.")
            
            # Create a grid of filters (max 3 per row)
            filter_cols = st.columns(min(len(other_sweeps), 3))
            
            for i, sweep_col in enumerate(other_sweeps):
                with filter_cols[i % 3]:
                    # Make a clean label for the UI (e.g., "Detector Det Eff D0")
                    clean_label = sweep_col.replace("sweep_", "").replace("_", " ").title()
                    
                    unique_vals = sorted(df[sweep_col].unique().tolist())
                    if not unique_vals:
                        continue
                        
                    selected_val = st.selectbox(
                        f"Lock {clean_label}", 
                        options=unique_vals,
                        key=f"lock_{sweep_col}",  # Unique key for Streamlit state
                        index=len(unique_vals) - 1  # Default to highest value
                    )
                    
                    # FILTER THE DATAFRAME to only this locked value
                    df = df[df[sweep_col] == selected_val]
    else:
        # Warn user if they selected "None" but multiple sweeps exist
        if len(sweep_columns) > 1:
            st.warning("⚠️ Multiple sweep parameters exist. Select one for 'Color/group' to separate the lines, otherwise they will overlap and zig-zag.")
    # ========================================================

    # --- CRITICAL FIX: SORT DATA BEFORE PLOTTING ---
    try:
        df = df.sort_values(by=[x_axis, color_axis] if color_axis != "None" else x_axis)
    except Exception:
        df = df.sort_values(by=x_axis)

    plot_df = df.copy()

    # --- PLOTTING LOGIC ---
    # Use log scale automatically for Key Rate and QBER
    use_log_y = any(kw in y_axis.lower() for kw in ["rate", "qber", "error"])

    if color_axis == "None":
        fig = px.line(
            plot_df, x=x_axis, y=y_axis, markers=True,
            title=f"{y_axis} vs {x_axis}",
            log_y=use_log_y
        )
    else:
        fig = px.line(
            plot_df, x=x_axis, y=y_axis, color=color_axis, markers=True,
            title=f"{y_axis} vs {x_axis} (Grouped by {color_axis})",
            log_y=use_log_y
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend_title_text=color_axis if color_axis != "None" else "",
        yaxis_title=y_axis + (" (log scale)" if use_log_y else ""),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Auto-generate QBER plot if applicable
    if qber_guess and qber_guess in numeric_columns and qber_guess != y_axis:
        st.markdown("#### QBER Plot")
        qber_fig = px.line(
            plot_df, x=x_axis, y=qber_guess, 
            color=None if color_axis == "None" else color_axis,
            markers=True, title=f"{qber_guess} vs {x_axis}"
        )
        qber_fig.update_layout(template="plotly_white", hovermode="x unified")
        st.plotly_chart(qber_fig, use_container_width=True)

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
        [path for path in run_dir.glob("**/*") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not files:
        st.info("There is no file to display.")
        return

    for path in files:
        relative = path.relative_to(run_dir)
        size_kb = path.stat().st_size / 1024

        with st.expander(f"{relative} — {size_kb:.1f} KB"):
            if path.suffix.lower() in [".json", ".csv", ".log", ".txt"]:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                    st.code(content[:20_000], language="json" if path.suffix.lower() == ".json" else "text")
                except Exception as exc:
                    st.warning(f"Cannot preview file: {exc}")

            st.download_button(
                label=f"Download {relative.name}",
                data=path.read_bytes(),
                file_name=relative.name,
                key=f"download-{path}",
            )


def render_help_panel() -> None:
    st.subheader("Integration Notes")

    st.markdown(
        """
        This dashboard calls `main_optimized.run_and_save_csv()` directly
        from Python — no subprocess or CLI bridge is needed.

        The simulation config dict and sweep dict are passed as native
        Python objects. Results are written to `qkd_results_optimized_sweep.csv`
        inside the run directory.

        **Key points:**

        1. **Config:** The base configuration is imported from
           `main_optimized.DEFAULT_CONFIG` — the UI always reflects the engine's
           true defaults.
        2. **Sweep:** The sweep dict is passed directly to `run_and_save_csv(sweeps=...)`.
        3. **Output:** The CSV file is automatically found in the run directory
           by the Results Explorer tab.
        """
    )

def main() -> None:
    # 1. Initialize page config and session state
    init_page()
    ensure_state()
    
    # 2. Render static elements (Header and Sidebar)
    render_header()
    output_dir, workers = render_sidebar()

    # 3. Create the main tab layout
    tab_setup, tab_execute, tab_results, tab_files, tab_help = st.tabs([
        "⚙️ Setup & Sweep", 
        "🚀 Execute Simulation", 
        "📊 Results Explorer", 
        "📁 Run Files", 
        "❓ Integration Notes"
    ])

    # 4. Populate Tab 1: Configuration and Sweeps
    with tab_setup:
        # We render the editors here and capture their return values
        current_config = render_config_editor()
        current_sweep = render_sweep_editor()
        
        # Store them in session state so the Execute tab can access them 
        # even if the user switches tabs without re-rendering the setup.
        st.session_state.config = current_config
        st.session_state.sweep = current_sweep

    # 5. Populate Tab 2: Execution
    with tab_execute:
        # Retrieve config and sweep from session state (fallback to defaults if tab wasn't visited)
        config_to_run = st.session_state.get("config", json.loads(json.dumps(DEFAULT_CONFIG)))
        sweep_to_run = st.session_state.get("sweep", json.loads(json.dumps(DEFAULT_SWEEP)))
        
        render_run_panel(
            config=config_to_run,
            sweep=sweep_to_run,
            output_dir=output_dir,
            workers=workers,
        )

    # 6. Populate Tab 3: Data Analysis
    with tab_results:
        render_results_panel()

    # 7. Populate Tab 4: File Browser & Logs
    with tab_files:
        render_files_panel()

    # 8. Populate Tab 5: Help/Documentation
    with tab_help:
        render_help_panel()


if __name__ == "__main__":
    main()
