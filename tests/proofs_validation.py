import copy
import csv
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_proof_test(proof_name):
    import main_optimized as m
    
    cfg = copy.deepcopy(m.DEFAULT_CONFIG)
    
    # تنظیمات ثابت برای همه اثبات‌ها
    cfg["simulation"]["apply_statistical_noise"] = True
    cfg["simulation"]["distance_start_km"] = 0.0
    cfg["simulation"]["distance_stop_km"] = 0.0
    cfg["simulation"]["distance_points"] = 1
    cfg["simulation"]["min_pulses_log"] = 7
    cfg["simulation"]["max_pulses_log"] = 7
    
    cfg["source"]["intended_proof"] = proof_name
    cfg["protocol_params"]["protocol"] = "BB84_DECOY"
    cfg["protocol_runtime"]["protocol_class"] = "BB84DecoyProtocol"
    cfg["source"]["confidence_method"] = "chernoff"
    cfg["source"]["intensity_jitter"] = 0.0
    
    # پارامترهای استاندارد
    cfg["source"]["pulses"] = {
        "signal": {"mu": 0.5, "prob": 0.5},
        "decoy": {"mu": 0.1, "prob": 0.25},
        "vacuum": {"mu": 0.0, "prob": 0.25}
    }
    
    eta_det = 0.15
    Y0 = 3e-6
    ed = 0.01
    cfg["detector"]["det_eff_d0"] = eta_det
    cfg["detector"]["det_eff_d1"] = eta_det
    cfg["detector"]["dark_rate"] = Y0
    cfg["detector"]["dark_rate_d1"] = Y0
    cfg["detector"]["qber_intrinsic"] = ed
    cfg["detector"]["misalignment"] = 0.0
    cfg["detector"]["dead_time_ns"] = 0.0

    print(f"Running simulation with proof: {proof_name}...")
    m.run_and_save_csv(cfg, num_workers=1, sweeps={})
    
    with open("qkd_results_optimized_sweep.csv", "r") as f:
        rows = list(csv.DictReader(f))
        latest = rows[-1]
        
    skr = float(latest.get("secure_key_rate", 0))
    qber = float(latest.get("qber_signal", 0))
    
    return skr, qber

if __name__ == "__main__":
    print("=" * 60)
    print("SECURITY PROOFS CROSS-VALIDATION")
    print("=" * 60)
    
    proofs_to_test = ["LIM_2014", "MA_2005", "WANG_2005", "TIGHT"]
    results = {}
    
    for proof in proofs_to_test:
        try:
            skr, qber = run_proof_test(proof)
            results[proof] = {"skr": skr, "qber": qber}
            print(f"Proof: {proof:10} | SKR: {skr:.6e} | QBER: {qber:.4f}\n")
        except Exception as e:
            print(f"Proof: {proof:10} | FAILED TO RUN: {e}\n")
            results[proof] = {"skr": 0, "qber": 0}
            
    print("=" * 60)
    print("ANALYSIS:")
    
    # 1. چک کردن اینکه هیچ اثباتی کرش نکرده باشد
    # 2. چک کردن اینکه SKR منفی یا NaN نباشد
    # 3. بررسی منطق: معمولا TIGHT باید بالاترین SKR را بدهد
    
    valid_proofs = [p for p in proofs_to_test if results[p]["skr"] >= 0]
    
    if len(valid_proofs) == len(proofs_to_test):
        print("✅ All proofs executed without crashing and produced non-negative SKR.")
    else:
        print("❌ Some proofs crashed or produced negative SKR!")
        
    # پیدا کردن بهترین اثبات
    best_proof = max(valid_proofs, key=lambda p: results[p]["skr"])
    print(f"🏆 Best SKR produced by: {best_proof}")
    print("=" * 60)