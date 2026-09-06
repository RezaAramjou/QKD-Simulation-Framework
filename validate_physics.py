import math
import sys
import copy
import csv

def channel_transmittance(L, alpha_dB_km):
    return 10 ** (-alpha_dB_km * L / 10)

def Q_mu_poisson(mu, eta, Y0):
    return 1 - (1 - Y0) * math.exp(-mu * eta)

def Q_mu_thermal(mu, eta, Y0, M=1):
    return 1 - (1 - Y0) / ((1 + mu * eta / M) ** M)

def E_mu_poisson(mu, eta, Y0, e_d):
    Q = Q_mu_poisson(mu, eta, Y0)
    if Q <= 0: return 0.5
    return (0.5 * Y0 + e_d * (Q - Y0)) / Q

def dead_time_correction(Q_ideal, tau_d_ns, T_pulse_ns):
    return Q_ideal / (1 + Q_ideal * (tau_d_ns / T_pulse_ns))

def validate_statistic(observed_count, total_trials, expected_prob, test_name):
    expected_count = total_trials * expected_prob
    variance = total_trials * expected_prob * (1 - expected_prob)
    if variance <= 0: variance = 1e-10
    std_dev = math.sqrt(variance)
    z_score = abs(observed_count - expected_count) / std_dev
    
    passed = z_score <= 3.8906
    print(f"{test_name:25} | Exp_p: {expected_prob:.6e} | Sim_p: {observed_count/total_trials:.6e} | Z: {z_score:6.2f} | {'PASS' if passed else 'FAIL'}")
    return passed

def run_isolated_simulation(config_overrides):
    import main_optimized as m
    cfg = copy.deepcopy(m.DEFAULT_CONFIG)
    
    N_pulses = 100000
    dist = config_overrides.get("dist", 0)
    
    cfg["simulation"]["apply_statistical_noise"] = True
    cfg["simulation"]["distance_start_km"] = dist
    cfg["simulation"]["distance_stop_km"] = dist
    cfg["simulation"]["distance_points"] = 1
    cfg["simulation"]["min_pulses_log"] = 5
    cfg["simulation"]["max_pulses_log"] = 5
    
    mu = config_overrides.get("mu", 0.5)
    cfg["source"]["statistics_type"] = config_overrides.get("stat_type", "POISSON")
    cfg["source"]["intensity_jitter"] = 0.0
    
    cfg["source"]["pulses"] = {
        "signal": {"mu": mu, "prob": 1.0},
        "decoy": {"mu": 0.1, "prob": 0.0},
        "vacuum": {"mu": 0.0, "prob": 0.0}
    }
    
    eta_det = config_overrides.get("det_eff", 0.15)
    Y0 = config_overrides.get("Y0", 3e-6)
    ed = config_overrides.get("ed", 0.01)
    
    cfg["detector"]["det_eff_d0"] = eta_det
    cfg["detector"]["det_eff_d1"] = eta_det
    cfg["detector"]["dark_rate"] = Y0
    cfg["detector"]["dark_rate_d1"] = Y0
    cfg["detector"]["qber_intrinsic"] = ed
    cfg["detector"]["misalignment"] = 0.0
    
    cfg["detector"]["dead_time_ns"] = config_overrides.get("dt", 0.0)
    cfg["detector"]["dead_time_model"] = "NON_PARALYZABLE"
    
    m.run_and_save_csv(cfg, num_workers=1, sweeps={})
    
    with open("qkd_results_optimized_sweep.csv", "r") as f:
        rows = list(csv.DictReader(f))
        latest = rows[-1]
        
    y_sig = float(latest.get("detection_yield_signal", 0))
    q_sig = float(latest.get("qber_signal", 0))
    t_pulse = float(cfg["source"]["pulse_period_ns"])
    
    return {
        "clicks": int(y_sig * N_pulses),
        "errors": int(q_sig * y_sig * N_pulses),
        "pulse_period": t_pulse
    }

if __name__ == "__main__":
    print("=" * 85)
    print("QKD PHYSICS VALIDATION - PURE STATE ISOLATION")
    print("Statistical Model: Binomial Z-Test (Alpha = 1e-4, Z_crit = 3.89)")
    print("=" * 85)
    
    N = 100000
    eta_det = 0.15
    Y0 = 3e-6
    ed = 0.01
    
    print("\n[TEST 1] Poisson WCP Asymptotic")
    for d in [0, 50, 100]:
        res = run_isolated_simulation({"dist": d, "mu": 0.5, "stat_type": "POISSON", "det_eff": eta_det, "Y0": Y0, "ed": ed, "dt": 0.0})
        eta = channel_transmittance(d, 0.2) * eta_det
        Q_th = Q_mu_poisson(0.5, eta, Y0) * 0.5
        E_th = E_mu_poisson(0.5, eta, Y0, ed)
        validate_statistic(res["clicks"], N, Q_th, f"WCP Gain L={d}km")
        validate_statistic(res["errors"], res["clicks"], E_th, f"WCP QBER L={d}km")

    print("\n[TEST 2] Thermal Single-Mode")
    res = run_isolated_simulation({"dist": 0, "mu": 0.5, "stat_type": "THERMAL", "det_eff": eta_det, "Y0": Y0, "dt": 0.0})
    Q_th = Q_mu_thermal(0.5, eta_det, Y0, M=1) * 0.5
    validate_statistic(res["clicks"], N, Q_th, "Thermal Gain L=0km")

    print("\n[TEST 3] Dead-Time Physics (Independent SPADs)")
    dt_ns = 50.0
    res = run_isolated_simulation({"dist": 0, "mu": 0.5, "stat_type": "POISSON", "det_eff": eta_det, "Y0": Y0, "dt": dt_ns})
    
    # فیزیک صحیح: محاسبه زمان مرده روی هر SPAD به طور مستقل
    Q_ideal_raw = Q_mu_poisson(0.5, eta_det, Y0)
    Q_per_spad = Q_ideal_raw / 2.0
    T_pulse = res["pulse_period"]
    
    Q_dead_per_spad = dead_time_correction(Q_per_spad, dt_ns, T_pulse)
    Q_dt_exp = Q_dead_per_spad * 2.0 * 0.5  # 2 SPADs * Sifting probability
    
    validate_statistic(res["clicks"], N, Q_dt_exp, f"Dead-time tau={dt_ns}ns")
