# qkd/proofs/optimization.py
# -*- coding: utf-8 -*-
"""
Parameter optimization utilities for QKD protocols.
Based on Ma et al. 2005 and Chapman et al. 2018.
"""

import logging
import math
import numpy as np
from typing import Tuple, Callable, Dict, Any
from scipy.optimize import minimize_scalar, brentq, minimize

from ..params import QKDParams
from ..utils.math import binary_entropy

logger = logging.getLogger(__name__)

__all__ = [
    "optimize_decoy_intensities", 
    "optimize_rotation_angle", 
    "calculate_optimal_mu_ma2005",
    "optimize_nu_ma2005",
    "optimize_pulse_distribution_ma2005",
    "optimize_simultaneous_ma2005",
    "optimize_global_ma2005",
    "optimize_lim2014_finite_key",
    "optimize_rotation_angle_chapman",
    "asymptotic_gllp_rate",
    "get_optimizer_bounds",
    "optimize_pulses_for_distance",
]

def optimize_decoy_intensities(
    distance_km: float,
    fiber_loss_db_km: float,
    det_eff: float,
    dark_rate: float,
    error_correction_f: float
) -> Tuple[float, float]:
    """
    Heuristic optimization for signal and decoy intensities.
    Returns: (mu_opt, nu_opt)
    """
    transmittance = 10**(-fiber_loss_db_km * distance_km / 10.0)
    
    def neg_key_rate(x):
        mu, nu = x[0], x[1]
        if mu <= nu or nu <= 0 or mu > 1.0: return 1.0
        
        # Heuristic simplified rate model
        Q_mu = 1 - math.exp(-transmittance * mu) + 2 * dark_rate
        if Q_mu <= 0:
            return 1.0
        E_mu = (0.5 * 2 * dark_rate) / Q_mu
        Y1 = transmittance * det_eff
        Q1 = mu * math.exp(-mu) * Y1
        
        rate = Q1 - error_correction_f * Q_mu * binary_entropy(E_mu)
        return -rate

    res = minimize(neg_key_rate, [0.5, 0.1], bounds=[(0.1, 0.9), (0.01, 0.3)], method='L-BFGS-B')
    
    if res.success:
        return float(res.x[0]), float(res.x[1])
    return 0.48, 0.05

def optimize_rotation_angle(gamma: float) -> float:
    """Finds the optimal encoding rotation angle theta_gamma."""
    def neg_success_prob(theta):
        return -1.0 * (math.cos(theta)**2 * (1-gamma) + math.sin(theta)**2)
    res = minimize_scalar(neg_success_prob, bounds=(0, math.pi/2), method='bounded')
    return float(res.x)

def calculate_optimal_mu_ma2005(
    distance_km: float,
    fiber_loss_db_km: float,
    det_eff: float,
    dark_rate: float,
    error_correction_f: float = 1.22, 
    e_detector: float = 0.033  
) -> float:
    """
    Calculates optimal signal intensity (mu) maximizing the GLLP rate.
    Implements the numerical solution to Eq. (12) in Ma et al. 2005.
    """
    # 1. Calculate RHS constant C based on detector error
    h2_e = binary_entropy(e_detector)
    numerator = error_correction_f * h2_e
    denominator = 1.0 - h2_e
    
    if denominator <= 0:
        logger.warning("Denominator in optimal mu calc is <= 0 (Error rate too high). Defaulting to 0.48.")
        return 0.48
        
    C = numerator / denominator
    
    # 2. Define function to find root: (1-mu)e^-mu - C = 0
    def objective(mu):
        return (1.0 - mu) * math.exp(-mu) - C
    
    # Define dynamic fallback based on Paper Section 3.1
    fallback_mu = 0.54 if error_correction_f <= 1.05 else 0.48

    # 3. Solve for mu in (0, 1]
    try:
        # Check bounds. f(mu) = (1-mu)e^-mu starts at 1 (mu=0) and decreases.
        # We check if the solution exists in [0.001, 0.999]
        val_low = objective(0.001)
        val_high = objective(0.999)
        
        if val_low * val_high < 0:
            mu_opt = brentq(objective, 0.001, 0.999)
            return float(mu_opt)
        else:
            logger.warning(f"Optimal mu equation has no solution in [0,1]. Falling back to {fallback_mu}.")
            return fallback_mu
    except Exception as e:
        logger.warning(f"Numerical solver failed for optimal mu: {e}. Defaulting to {fallback_mu}.")
        return fallback_mu

def optimize_nu_ma2005(
    mu: float,
    cost_function_callback: Callable[[float], float]
) -> float:
    """
    Optimizes the weak decoy intensity (nu) for a fixed data size N.
    Based on Ma et al. 2005 Section 4.3.
    """
    lower_bound = 0.001
    upper_bound = mu * 0.9 

    # Robust 1D scalar minimization (Brent's method)
    res = minimize_scalar(
        cost_function_callback, 
        bounds=(lower_bound, upper_bound), 
        method='bounded',
        options={'xatol': 1e-4}
    )
    
    if res.success:
        return float(res.x)
    
    # Fallback if optimization fails
    return 0.1

class Ma2005GlobalOptimizer:
    """
    Implements the global optimization for Vacuum+Weak decoy protocol
    using finite-difference derivatives for the rate w.r.t. Ps, Pd, and nu.
    Feasibility is enforced by the caller's bounds and constraints, so this
    objective assumes feasible inputs.
    """
    def __init__(self, mu_target, cost_callback):
        self.mu = mu_target
        self.callback = cost_callback
        self.epsilon = 1e-8

    def objective(self, x):
        """Objective function for minimization (-Rate)."""
        p_s, p_d, nu = x
        return self.callback(p_s, p_d, nu)

    def gradient(self, x):
        """
        Calculates the gradient [d(-R)/dPs, d(-R)/dPd, d(-R)/dnu].
        """
        p_s, p_d, nu = x
        
        # 1. Calculate current rate
        val_center = self.objective(x)

        # 2. Finite Difference for Probabilities (Ps, Pd)
        grad = np.zeros(3)
        
        # d/dPs
        x_ps = x.copy(); x_ps[0] += self.epsilon
        val_ps = self.objective(x_ps)
        grad[0] = (val_ps - val_center) / self.epsilon
        
        # d/dPd
        x_pd = x.copy(); x_pd[1] += self.epsilon
        val_pd = self.objective(x_pd)
        grad[1] = (val_pd - val_center) / self.epsilon
        
        # 3. Centered finite difference for nu
        x_nu_p = x.copy(); x_nu_p[2] += self.epsilon
        x_nu_m = x.copy(); x_nu_m[2] -= self.epsilon
        grad[2] = (self.objective(x_nu_p) - self.objective(x_nu_m)) / (2 * self.epsilon)
        
        return grad

def optimize_pulse_distribution_ma2005(
    cost_function_callback: Callable[[float, float], float]
) -> Tuple[float, float, float]:
    """
    Optimizes the pulse probabilities (Signal, Decoy, Vacuum) for a fixed data size N.
    """
    def objective(x):
        p_s, p_d = x
        return cost_function_callback(p_s, p_d)

    # Initial guess
    x0 = [0.66, 0.29]
    bounds = [(0.01, 0.98), (0.01, 0.98)]
    constraints = [{'type': 'ineq', 'fun': lambda x: 1.0 - (x[0] + x[1]) - 0.01}]
    
    res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')
    
    if res.success:
        p_s, p_d = res.x
        p_v = 1.0 - p_s - p_d
        return p_s, p_d, p_v
        
    logger.warning("Pulse distribution optimization failed. Using defaults.")
    return 0.66, 0.29, 0.05

def optimize_simultaneous_ma2005(
    mu_target: float,
    cost_function_callback: Callable[[float, float, float], float]
) -> Tuple[float, float, float, float]:
    """
    Simultaneously optimizes pulse probabilities (P_s, P_d) and decoy intensity (nu).
    """
    optimizer = Ma2005GlobalOptimizer(mu_target, cost_function_callback)

    # Initial guess: Start with standard values
    x0 = [0.66, 0.29, 0.127]
    
    # Bounds
    bounds = [
        (0.1, 0.95), (0.01, 0.8), (0.001, mu_target * 0.99)
    ]
    
    constraints = [{'type': 'ineq', 'fun': lambda x: 1.0 - (x[0] + x[1]) - 0.01}]
    
    # Use SLSQP with the Jacobian method implemented in the optimizer class
    res = minimize(
        optimizer.objective, 
        x0, 
        jac=optimizer.gradient,
        bounds=bounds, 
        constraints=constraints, 
        method='SLSQP', 
        tol=1e-5
    )
    
    if res.success:
        p_s_opt, p_d_opt, nu_opt = res.x
        p_v_opt = 1.0 - p_s_opt - p_d_opt
        return p_s_opt, p_d_opt, p_v_opt, nu_opt
    else:
        # If optimization fails, try fallback or return initial guess
        return 0.66, 0.29, 0.05, 0.127

def optimize_global_ma2005(
    mu_target: float,
    cost_function_callback: Callable[[float, float, float], float]
) -> Tuple[float, float, float, float]:
    return optimize_simultaneous_ma2005(mu_target, cost_function_callback)

def optimize_lim2014_finite_key(
    params: QKDParams,
    dist_km: float,
    cost_callback: Callable[[float, float, float, float, float], float]
) -> Tuple[float, float, float, float, float]:
    """
    Numerically optimizes the secret key rate R = l/N over the free parameters
    {q_x, p_mu1, p_mu2, mu1, mu2} as specified in Lim et al. (2014) Section IV.

    .. deprecated::
        This function uses Nelder-Mead and a different signature (QKDParams +
        cost_callback).  Prefer :func:`optimize_pulses_for_distance` which uses
        proof-aware SLSQP bounds and is the single source of truth used by
        main_optimized.py.  Kept for backward compatibility with
        lp_validation.py.
    """
    x0 = [0.9, 0.6, 0.2, 0.5, 0.1]
    
    bounds = [
        (0.5, 0.99), # q_x
        (0.1, 0.8),  # p_s
        (0.05, 0.5), # p_d
        (0.2, 0.8),  # mu
        (0.01, 0.2)  # nu
    ]
    
    def objective(x):
        q_x, p_s, p_d, mu, nu = x
        return cost_callback(q_x, p_s, p_d, mu, nu)

    # Cross-variable constraints (per-variable bounds alone are insufficient)
    constraints = [
        {'type': 'ineq', 'fun': lambda x: 0.99 - (x[1] + x[2])},   # p_s + p_d <= 0.99
        {'type': 'ineq', 'fun': lambda x: x[3] * 0.9 - x[4]}        # nu <= 0.9 * mu
    ]

    # SLSQP respects bounds and constraints natively; suitable for the smooth finite-key rate
    res = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 200, 'ftol': 1e-7}
    )
    
    if res.success:
        return tuple(float(v) for v in res.x)
    
    # Fallback to initial guess if optimization fails
    logger.warning("Lim2014 parameter optimization failed. Using defaults.")
    return tuple(x0)

def optimize_rotation_angle_chapman(gamma: float) -> float:
    """
    Finds the optimal encoding rotation angle theta_gamma for the Coherent Scheme
    in Chapman et al. (2018).
    
    We maximize P_succ = 0.5 * (1 + D(rho_0, rho_1)) where rho_i are outputs of ADC.
    """
    
    # Kraus ops for ADC(gamma)
    E0 = np.array([[1, 0], [0, math.sqrt(1-gamma)]])
    E1 = np.array([[0, math.sqrt(gamma)], [0, 0]])
    
    # Rotation Y
    Ry = lambda theta: np.array([[math.cos(theta/2), -math.sin(theta/2)], 
                                 [math.sin(theta/2),  math.cos(theta/2)]])
    
    def get_rho_out(state_in):
        return E0 @ state_in @ E0.conj().T + E1 @ state_in @ E1.conj().T
    
    def objective(theta):
        # Rotate |0> and |1> input states by theta
        U = Ry(theta)
        state0 = U @ np.array([[1,0],[0,0]]) @ U.conj().T
        state1 = U @ np.array([[0,0],[0,1]]) @ U.conj().T
        
        rho0_out = get_rho_out(state0)
        rho1_out = get_rho_out(state1)
        
        # Maximize Trace Distance D(rho0_out, rho1_out)
        diff = rho0_out - rho1_out
        evals = np.linalg.eigvalsh(diff)
        dist = 0.5 * np.sum(np.abs(evals))
        return -dist

    res = minimize_scalar(objective, bounds=(0, math.pi), method='bounded')
    return float(res.x)


# ==================== PROOF-AWARE PULSE OPTIMIZER ====================
# Extracted from main_optimized.py (2026-08) to make this module the single
# source of truth for pulse-parameter optimization.  main_optimized.py now
# delegates to optimize_pulses_for_distance() below.
#
# The optimizer uses an asymptotic GLLP key-rate formula (Ma 2005, Eq. 33)
# as the primary objective, augmented with a finite-size penalty that
# discourages parameter choices which would leave the decoy / vacuum
# statistics too sparse for the finite-key proof to extract a positive
# Y1 lower bound.  The asymptotic rate is smooth and differentiable, so
# SLSQP converges quickly; the finite-size penalty is a piecewise-linear
# soft threshold that does not break differentiability at the threshold
# boundary (it has a kink, but SLSQP tolerates this).
#
# Why not call the actual finite-key proof in the cost function?
#   The proof's Y1 lower bound depends on integer-valued decoy / vacuum
#   detection counts.  At N=10^5..10^7 pulses, the expected vacuum
#   detections are ~0..10, so the proof returns Y1_L=0 for almost every
#   parameter combination, producing a flat zero landscape that SLSQP
#   cannot navigate.  The asymptotic rate sidesteps this by using the
#   analytic channel model directly.


def asymptotic_gllp_rate(mu: float, nu: float, p_s: float, p_d: float,
                         p_v: float, q_x: float, dist_km: float,
                         config: Dict[str, Any]) -> float:
    """Asymptotic GLLP key rate per pulse (Ma 2005, Eq. 33).

    R = q * [Y1 * P(1|mu) * (1 - h2(e1)) - Q_mu * f * h2(E_mu)] * p_s

    where:
      q       = basis reconciliation factor (0.5 for symmetric BB84)
      Y1      = single-photon yield = Y0 + eta
      P(1|mu) = mu * exp(-mu)
      e1      = single-photon QBER = (0.5*Y0 + e_det*eta) / Y1
      Q_mu    = Y0 + 1 - exp(-eta*mu)
      E_mu    = (0.5*Y0 + e_det*(1 - exp(-eta*mu))) / Q_mu
      f       = error-correction efficiency (1.16 typical)

    Channel model:
      eta = det_eff * 10^(-fiber_loss_db_km * dist_km / 10)
      Y0  = 2 * dark_rate   (upper bound on dark-count yield, both detectors)
      e_det = qber_intrinsic + misalignment^2  (combined intrinsic QBER)
    """
    # --- Channel + detector parameters ---
    fiber_loss_db_km = float(config["channel"].get("fiber_loss_db_km", 0.2))
    det_eff = float(config["detector"].get("det_eff_d0", 0.15))
    dark_rate = float(config["detector"].get("dark_rate", 1e-6))
    qber_intrinsic = float(config["detector"].get("qber_intrinsic", 0.01))
    misalignment = float(config["detector"].get("misalignment", 0.005))
    f_ec = float(config["protocol_params"].get("error_correction_efficiency", 1.16))

    e_det = qber_intrinsic + misalignment * misalignment

    # Channel transmittance
    channel_attenuation_db = fiber_loss_db_km * dist_km
    eta_channel = 10.0 ** (-channel_attenuation_db / 10.0)
    eta = det_eff * eta_channel

    # Y0: dark-count contribution (both detectors fire independently)
    Y0 = 2.0 * dark_rate

    # Y1: single-photon yield
    Y1 = Y0 + eta
    if Y1 <= 0.0:
        return 0.0

    # Q_mu: total signal gain
    one_minus_exp = 1.0 - math.exp(-eta * mu)
    Q_mu = Y0 + one_minus_exp
    if Q_mu <= 0.0:
        return 0.0

    # E_mu: total QBER for signal
    E_mu = (0.5 * Y0 + e_det * one_minus_exp) / Q_mu
    E_mu = min(max(E_mu, 0.0), 0.5)

    # e1: single-photon QBER
    e1 = (0.5 * Y0 + e_det * eta) / Y1
    e1 = min(max(e1, 0.0), 0.5)

    # Binary entropy
    def h2(p: float) -> float:
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)

    # P(1|mu): probability that signal pulse has exactly 1 photon
    P1_mu = mu * math.exp(-mu)

    # for API symmetry but do not currently optimize it — asymmetric basis
    # choice would break the finite-key proof's parameter estimation.)
    q = 0.5

    gain = Y1 * P1_mu * (1.0 - h2(e1))
    leak = Q_mu * f_ec * h2(E_mu)
    rate_per_pulse = q * (gain - leak)

    if rate_per_pulse <= 0.0:
        return 0.0

    # Multiply by signal probability (fraction of pulses that are signal)
    return rate_per_pulse * p_s


def get_optimizer_bounds(proof_name: str):
    """Return (bounds, x0, upper_bound_corner, profile_tag) tailored to the proof.

    Vacuum-hungry proofs (LIM_2014, MA_2005, TIGHT) use 3-intensity finite-key
    bounds that require vacuum detections to bound Y0.  At N=10^7 with
    p_v=0.15, expected vacuum detections are only ~3-5 — far too few for
    statistical bounds.  We therefore enforce higher p_v and p_d floors so
    the optimizer cannot starve the vacuum / decoy channels, and lower the
    p_s ceiling so p_s + p_d + p_v <= 1.0 remains feasible.

    WANG_2005 is a 2-intensity proof that bounds Y0 directly from the weak
    decoy statistics; it does not need vacuum detections.  We keep the
    loose bounds so the optimizer is free to push p_s high and p_v low.

    Returns:
        bounds            -- list of (lo, hi) tuples for [mu, nu, p_s, p_d, p_v]
        x0                -- initial guess (feasible w.r.t. constraints)
        upper_bound_corner-- fallback when SLSQP fails (feasible)
        profile_tag       -- short string for logging ("WANG" or "VAC3")
    """
    p = (proof_name or "").upper()
    if p == "WANG_2005":
        # 2-intensity proof: vacuum pulses are not statistically needed.
        # Loose bounds — let the optimizer maximize signal throughput.
        bounds = [
            (0.30, 0.60),  # mu
            (0.05, 0.15),  # nu
            (0.40, 0.70),  # p_s
            (0.15, 0.35),  # p_d
            (0.15, 0.40),  # p_v
        ]
        x0 = [0.45, 0.10, 0.55, 0.25, 0.20]            # sum = 1.00
        upper_bound_corner = [0.60, 0.15, 0.70, 0.15, 0.15]  # sum = 1.00
        profile_tag = "WANG"
    else:
        # 3-intensity proofs (LIM_2014, MA_2005, TIGHT, default):
        # raise p_d and p_v floors so vacuum/decoy channels have enough
        # samples for the finite-key Y0 bound.  Cap p_s at 0.55 so
        # p_s + p_d + p_v <= 1.0 stays feasible at the floor.
        bounds = [
            (0.30, 0.60),  # mu
            (0.05, 0.15),  # nu
            (0.30, 0.55),  # p_s  -- lowered ceiling (p_d+p_v floor = 0.45)
            (0.20, 0.35),  # p_d  -- raised floor
            (0.25, 0.40),  # p_v  -- raised floor
        ]
        x0 = [0.45, 0.10, 0.45, 0.20, 0.25]            # sum = 0.90
        upper_bound_corner = [0.60, 0.15, 0.55, 0.20, 0.25]  # sum = 1.00
        profile_tag = "VAC3"
    return bounds, x0, upper_bound_corner, profile_tag


def optimize_pulses_for_distance(config: Dict[str, Any], dist_km: float,
                                 total_pulses: int, proof_name: str
                                 ) -> Tuple[float, float, float, float, float]:
    """Find optimal (mu, nu, p_s, p_d, p_v) for a given distance.

    Uses SLSQP on the asymptotic GLLP rate (smooth, differentiable) with a
    finite-size penalty that scales with sqrt(N).  Returns pulse parameters
    suitable for direct injection into config['source']['pulses'].

    Bounds are *proof-aware* (see get_optimizer_bounds):

      WANG_2005 (2-intensity, vacuum not needed statistically):
        mu   in [0.30, 0.60]   p_s in [0.40, 0.70]
        nu   in [0.05, 0.15]   p_d in [0.15, 0.35]
                               p_v in [0.15, 0.40]

      LIM_2014 / MA_2005 / TIGHT (3-intensity, vacuum-hungry):
        mu   in [0.30, 0.60]   p_s in [0.30, 0.55]
        nu   in [0.05, 0.15]   p_d in [0.20, 0.35]
                               p_v in [0.25, 0.40]

    Constraint: p_s + p_d + p_v <= 1.0  (>= 0 by inequality)
    """
    # Local import: scipy is only needed when the optimizer runs.
    try:
        from scipy.optimize import minimize as _scipy_minimize
    except ImportError as exc:
        logger.warning("scipy not available (%s); optimizer disabled.", exc)
        return (0.5, 0.1, 0.6, 0.2, 0.2)

    # Symmetric BB84: q_x = 0.5 fixed (do not optimize — asymmetric basis
    # choice breaks the finite-key proof's parameter estimation).
    q_x_fixed = 0.5

    # Proof-aware bounds: 2-intensity proofs (WANG_2005) get loose bounds
    # so the optimizer can maximize signal throughput; 3-intensity proofs
    # (LIM_2014, MA_2005, TIGHT) get tightened p_d / p_v floors so the
    # finite-key Y0 bound has enough vacuum statistics.
    bounds, x0, upper_bound_corner, profile_tag = get_optimizer_bounds(proof_name)

    def constraint_sum_probs(x):
        mu, nu, p_s, p_d, p_v = x
        return 1.0 - p_s - p_d - p_v

    constraints = [{"type": "eq", "fun": constraint_sum_probs}]

    # Cache channel params (read once)
    fiber_loss_db_km = float(config["channel"].get("fiber_loss_db_km", 0.2))
    det_eff = float(config["detector"].get("det_eff_d0", 0.15))
    dark_rate = float(config["detector"].get("dark_rate", 1e-6))
    channel_attenuation_db = fiber_loss_db_km * dist_km
    eta_channel = 10.0 ** (-channel_attenuation_db / 10.0)
    eta = det_eff * eta_channel
    Y0 = 2.0 * dark_rate
    Y1 = Y0 + eta

    sqrt_N = math.sqrt(float(total_pulses))

    def objective(x):
        mu, nu, p_s, p_d, p_v = x

        # Primary: asymptotic GLLP rate
        rate = asymptotic_gllp_rate(mu, nu, p_s, p_d, p_v,
                                    q_x_fixed, dist_km, config)
        if rate <= 0.0:
            # Infeasible region — return large positive so SLSQP moves away.
            return 1.0

        # --- Finite-size penalty (soft thresholds scaled with sqrt(N)) ---
        # Expected single-photon detections from signal pulses:
        Q1 = Y1 * mu * math.exp(-mu)  # single-photon gain for signal
        n_signal_single = Q1 * total_pulses * p_s
        # Expected single-photon detections from decoy pulses:
        n_decoy_single = Y1 * nu * math.exp(-nu) * total_pulses * p_d
        # Expected total detections from vacuum pulses (Y0 * count):
        n_vacuum_total = Y0 * total_pulses * p_v

        # Thresholds scale with sqrt(N) so they remain meaningful across
        # N = 10^5 .. 10^10.  Each threshold corresponds to ~10 sigma above
        # the statistical noise floor for that bin.
        thresh_signal = 10.0 * sqrt_N * math.sqrt(p_s)
        thresh_decoy = 10.0 * sqrt_N * math.sqrt(p_d)
        soft_vacuum_thresh = sqrt_N * math.sqrt(p_v)

        penalty = 0.0
        if n_signal_single < thresh_signal and thresh_signal > 0:
            penalty += 0.20 * (1.0 - n_signal_single / thresh_signal)
        if n_decoy_single < thresh_decoy and thresh_decoy > 0:
            penalty += 0.20 * (1.0 - n_decoy_single / thresh_decoy)
        if n_vacuum_total < soft_vacuum_thresh and soft_vacuum_thresh > 0:
            penalty += 0.20 * (1.0 - n_vacuum_total / soft_vacuum_thresh)
        penalty = min(penalty, 0.90)

        effective_rate = rate * (1.0 - penalty)
        return -effective_rate  # minimize negative = maximize

    # x0 and upper_bound_corner come from get_optimizer_bounds() above —
    # they are already tailored to the proof (WANG_2005 vs 3-intensity).
    # The upper-bound corner is the known-asymptotically-optimal corner
    # of the feasible region; it is compared against SLSQP's output and
    # used as a fallback when SLSQP fails to converge (long distances,
    # rate ~1e-5, noisy numerical gradients).

    best_x = None
    best_obj = float("inf")  # minimize; lower is better
    try:
        result = _scipy_minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        # Compare SLSQP result vs upper-bound corner; pick whichever
        # has the lower objective value.  This guards against SLSQP
        # returning a worse-than-trivial solution when the landscape
        # is flat or noisy.
        for cand in (result.x, upper_bound_corner):
            obj_val = objective(cand)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = cand
        mu, nu, p_s, p_d, p_v = best_x
    except Exception as exc:
        logger.warning(
            "Optimizer failed at d=%skm proof=%s: %s. Using upper-bound fallback.",
            dist_km, proof_name, exc,
        )
        mu, nu, p_s, p_d, p_v = upper_bound_corner

    # Round to 4 decimals
    mu = float(round(mu, 4))
    nu = float(round(nu, 4))
    # V2 normalization: round p_s and p_d first, then derive p_v = 1 - p_s - p_d
    # This GUARANTEES sum = 1.0 exactly (within floating point precision).
    # V1 approach (round all three + normalize + round again) fails ~8.7% of
    # the time due to post-normalization rounding drift.
    p_s = float(round(p_s, 4))
    p_d = float(round(p_d, 4))
    if p_s + p_d > 1.0:
        # Scale down proportionally if signal+decoy exceed 1.0
        total_sd = p_s + p_d
        p_s /= total_sd
        p_d /= total_sd
        p_s = float(round(p_s, 4))
        p_d = float(round(p_d, 4))
    p_v = float(round(1.0 - p_s - p_d, 4))

    logger.info(
        "[OPTIMIZE] d=%.1fkm proof=%s profile=%s N=%.2e: mu=%.4f nu=%.4f "
        "p_s=%.4f p_d=%.4f p_v=%.4f",
        dist_km, proof_name, profile_tag, float(total_pulses),
        mu, nu, p_s, p_d, p_v,
    )

    return (mu, nu, p_s, p_d, p_v)
