import sys
import os
import math

# تنظیم مسیر پروژه
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from qkd.proofs.optimization import optimize_lim2014_finite_key
from qkd.params import load_lim2014_dedicated_params

def H2(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def analytical_skr(q_x, p_s, p_d, mu, nu, eta, Y0, ed, f_ec):
    """تابع هزینه تحلیلی برای بررسی پارامترها (اصلاح شده)"""
    if p_s + p_d >= 1.0: return 0.0
    if nu >= mu * 0.9: return 0.0
    
    Q_mu = 1 - (1 - Y0) * math.exp(-eta * mu)
    E_mu = (0.5 * Y0 + ed * (1 - math.exp(-eta * mu))) / Q_mu
    Q_nu = 1 - (1 - Y0) * math.exp(-eta * nu)
    
    if abs(mu - nu) < 1e-9: return 0.0
        
    # اصلاح فرمول Lo2005 برای Y1
    term1 = Q_nu * math.exp(nu)
    term2 = (nu**2 / mu**2) * Q_mu * math.exp(mu)
    factor = mu / (mu * nu - nu**2)
    Y1_L = factor * (term1 - term2)
    
    Y1 = max(Y1_L, Y0)  # Y1 نمی‌تواند کمتر از نویز تاریک باشد
    e1 = (0.5 * Y0 + ed * eta) / Y1 if Y1 > 0 else 0.5
    
    Q1 = mu * math.exp(-mu) * Y1
    R = q_x * (Q1 * (1 - H2(e1)) - Q_mu * f_ec * H2(E_mu))
    return max(R, 0.0)

def run_lp_test(dist_km):
    print(f"\n[TEST] LP Solver at L={dist_km} km")
    eta_det = 0.15
    Y0 = 3e-6
    ed = 0.01
    f_ec = 1.16
    alpha = 0.2
    eta = 10 ** (-alpha * dist_km / 10) * eta_det
    
    params = load_lim2014_dedicated_params(distance_km=dist_km, num_bits=10**7)
    
    def cost_callback(q_x, p_s, p_d, mu, nu):
        # بهینه‌ساز به دنبال مینیمم کردن هزینه است، پس منفی SKR را برمی‌گردانیم
        skr = analytical_skr(q_x, p_s, p_d, mu, nu, eta, Y0, ed, f_ec)
        return -skr
        
    # ۱. محاسبه SKR با پارامترهای ثابت و سخت‌کد شده
    R_hardcoded = analytical_skr(0.5, 0.5, 0.25, 0.5, 0.1, eta, Y0, ed, f_ec)
    
    # ۲. اجرای LP Solver فریم‌ورک
    try:
        q_x_opt, p_s_opt, p_d_opt, mu_opt, nu_opt = optimize_lim2014_finite_key(params, dist_km, cost_callback)
        R_optimized = analytical_skr(q_x_opt, p_s_opt, p_d_opt, mu_opt, nu_opt, eta, Y0, ed, f_ec)
        
        print(f"  Hardcoded (mu=0.5, nu=0.1)     -> SKR: {R_hardcoded:.6e}")
        print(f"  Optimized  (mu={mu_opt:.3f}, nu={nu_opt:.3f}) -> SKR: {R_optimized:.6e}")
        
        if R_optimized >= R_hardcoded - 1e-9:
            print("  ✅ PASS: LP Solver successfully found optimal or equal parameters.")
            return True
        else:
            print("  ❌ FAIL: LP Solver performed worse than hardcoded parameters!")
            return False
            
    except Exception as e:
        import traceback
        print("  ❌ FAIL: LP Solver crashed!")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("LP SOLVER (PARAMETER OPTIMIZATION) VALIDATION")
    print("=" * 60)
    
    # تست در فواصل کوتاه، متوسط و طولانی
    res1 = run_lp_test(0.0)
    res2 = run_lp_test(50.0)
    res3 = run_lp_test(100.0)
    
    print("\n" + "=" * 60)
    if res1 and res2 and res3:
        print("🎉 100% LP SOLVER VALIDATION PASSED!  🎉")
    else:
        print("❌ LP SOLVER VALIDATION FAILED.")
    print("=" * 60)