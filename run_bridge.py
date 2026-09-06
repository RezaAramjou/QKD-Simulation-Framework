# run_bridge.py
import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Bridge between QKD UI and main_optimized.py")
    parser.add_argument("--config", required=True, help="Path to base config JSON from UI")
    parser.add_argument("--sweep", required=True, help="Path to sweep JSON from UI")
    parser.add_argument("--output-dir", required=True, help="Directory to save CSV outputs")
    parser.add_argument("--workers", type=int, default=1, help="Number of multiprocessing workers")
    
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    with open(args.sweep, 'r', encoding='utf-8') as f:
        sweep = json.load(f)

    SCRIPT_DIR = Path(__file__).parent.resolve()
    MAIN_SCRIPT = SCRIPT_DIR / "main_optimized.py"

    sim_cfg = config.get("simulation", {})
    det_cfg = config.get("detector", {})
    ch_cfg = config.get("channel", {})
    proto_cfg = config.get("protocol_runtime", {})
    proto_params_cfg = config.get("protocol_params", {})

    # ========================================================
    # FIX: Properly extract distance settings from UI config
    # ========================================================
    # If the UI provides simulation distance settings, use them.
    # Otherwise, fall back to a safe default.
    distance_start = sim_cfg.get("distance_start_km", 0.0)
    distance_stop = sim_cfg.get("distance_stop_km", 100.0)
    distance_points = sim_cfg.get("distance_points", 11) # <--- NOW READS FROM UI!
    
    # ========================================================

    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--protocol-class", proto_cfg.get("protocol_class", "BB84DecoyProtocol"),
        "--fiber-loss", str(ch_cfg.get("fiber_loss_db_km", 0.2)),
        "--det-eff-d0", str(det_cfg.get("det_eff_d0", 0.15)),
        "--det-eff-d1", str(det_cfg.get("det_eff_d1", 0.15)),
        "--dark-rate", str(det_cfg.get("dark_rate", 1e-6)),
        "--qber-intrinsic", str(det_cfg.get("qber_intrinsic", 0.01)),
        "--misalignment", str(det_cfg.get("misalignment", 0.005)),
        "--detector-type", det_cfg.get("detector_type", "SPD"),
        "--min-pulses-log", str(sim_cfg.get("min_pulses_log", 4)),
        "--max-pulses-log", str(sim_cfg.get("max_pulses_log", 6)),
        "--rng-seed", str(sim_cfg.get("rng_seed", 12345)),
        "--distance-start-km", str(distance_start),
        "--distance-stop-km", str(distance_stop),
        "--distance-points", str(distance_points), # <--- PASSED CORRECTLY!
        "--workers", str(args.workers),
        "--sweep-json", json.dumps(sweep),
    ]

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting simulation in: {run_dir}")
    print(f"Distance range: {distance_start} km to {distance_stop} km ({distance_points} points)")
    print(f"Command: {' '.join(cmd)}\n")
    
    import subprocess
    process = subprocess.Popen(cmd, cwd=run_dir)
    process.wait()
    
    sys.exit(process.returncode)

if __name__ == "__main__":
    main()
