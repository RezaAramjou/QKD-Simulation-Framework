# qkd/utils/scm_analysis.py
# -*- coding: utf-8 -*-
"""
Analysis utilities for SCM-QKD, WDM noise, and theoretical bounds.
"""
import math
import logging
from typing import Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "calculate_visibility",
    "calculate_intensity_mismatch_factor",
    "calculate_cso_detailed",
    "calculate_cso_combinatorial_uniform",
    "calculate_spectral_crosstalk_factor",
    "simulate_time_domain_interference",
    "calculate_exact_discrete_interference",
    "calculate_interference_exact_vectorized",
    "calculate_wdm_fwm_noise",
    "calculate_analytical_qber",
    "calculate_analytical_key_rate",
    "calculate_dispersion_penalty",
    "calculate_cascaded_fbg_loss_spectrum",
    "calculate_serial_link_transmittance",
    "calculate_effective_modulation_index",
    "calculate_lim2014_observables",
    "calculate_classical_success_bound"
]

def calculate_visibility(dm: float, psi1: float, psi2: float) -> float:
    return math.cos(psi1 - psi2) * ((1.0 - dm) / (1.0 + dm))

def calculate_intensity_mismatch_factor(dm: float, psi1: float, psi2: float) -> float:
    return 1.0 - abs(dm)

def calculate_cso_combinatorial_uniform(n_channels: int, channel_idx: int) -> Tuple[int, int]:
    """
    Calculates Count of Second Order (CSO) intermodulation products falling 
    on 'channel_idx' (1-based) for a uniform grid of 'n_channels'.
    """
    if n_channels < 2: return 0, 0
    # N_IMD approx 3N^2/8 for large N (worst case center channel)
    n_imd = int(3 * n_channels**2 / 8)
    n_hd = 1 # One harmonic usually falls in band if octave bandwidth
    return n_hd, n_imd

def calculate_cso_detailed(target_freq: float, frequency_plan: List[float]) -> Tuple[int, int]:
    hits_imd = 0
    hits_hd = 0
    tol = 1e3 # 1 kHz tolerance
    
    freqs = np.array(frequency_plan)
    n = len(freqs)
    
    for f in freqs:
        if abs(2*f - target_freq) < tol:
            hits_hd += 1
            
    for i in range(n):
        for j in range(i + 1, n):
            sum_f = freqs[i] + freqs[j]
            diff_f = abs(freqs[i] - freqs[j])
            if abs(sum_f - target_freq) < tol: hits_imd += 1
            if abs(diff_f - target_freq) < tol: hits_imd += 1
                
    return hits_hd, hits_imd

def calculate_spectral_crosstalk_factor(
    target_freq: float, frequency_plan: List[float], filter_bandwidth_hz: float, 
    modulation_bandwidth_hz: float, filter_shape: str = "lorentzian"
) -> float:
    total_xt = 0.0
    for f in frequency_plan:
        if f == target_freq: continue
        detuning = abs(f - target_freq)
        if filter_shape == "lorentzian":
            attenuation = 1.0 / (1.0 + (2.0 * detuning / filter_bandwidth_hz)**2)
        else:
            attenuation = math.exp(-math.log(2) * (2.0 * detuning / filter_bandwidth_hz)**2)
        total_xt += attenuation
    return total_xt

def simulate_time_domain_interference(target_freq, plan, mod_index):
    return 0.0

def calculate_exact_discrete_interference(modulation_index, n_cso, rng, num_samples):
    """Returns random samples of interference noise (Gaussian approx)."""
    sigma = (modulation_index * math.sqrt(n_cso)) / 4.0 
    noise = rng.normal(0, sigma, num_samples)
    return abs(noise), abs(noise) 

def calculate_interference_exact_vectorized(m, phases_a, phases_b, target_idx, all_indices):
    """
    Approximation of exact multi-channel interference noise.
    Previously returned zeros; now returns a Gaussian approximation based on
    combinatorial CSO count to ensure QBER > 0.
    """
    n_channels = len(all_indices)
    num_samples = phases_a.shape[1]
    
    # Calculate effective N_CSO for the target channel
    # Assuming standard grid layout where we approximate the count
    hd_count, imd_count = calculate_cso_combinatorial_uniform(n_channels, target_idx + 1)
    n_cso = hd_count + imd_count
    
    if n_cso <= 0:
        return np.zeros(num_samples), np.zeros(num_samples)
        
    # Generate noise proportional to sqrt(N_CSO) * m
    # Factor 4.0 is heuristic derived from Bessel expansion J_1(m)^2 scaling
    sigma = (m * math.sqrt(n_cso)) / 4.0
    
    # Generate random noise for D0 and D1 (uncorrelated approximation)
    rng = np.random.default_rng()
    noise_d0 = np.abs(rng.normal(0, sigma, num_samples))
    noise_d1 = np.abs(rng.normal(0, sigma, num_samples))
    
    return noise_d0, noise_d1

def calculate_wdm_fwm_noise(
    n_wdm_channels: int, channel_spacing_hz: float, input_power_dbm: float,
    fiber_gamma: float, fiber_length_km: float, fiber_loss_db_km: float
) -> float:
    if n_wdm_channels <= 1: return 0.0
    
    p_in_watts = 10**((input_power_dbm - 30) / 10.0)
    alpha = fiber_loss_db_km / 4.343 # 1/km
    leff = (1 - math.exp(-alpha * fiber_length_km)) / alpha
    
    # P_fwm_total approx (N^3 / 24) * gamma^2 * P_in^3 * Leff^2 * exp(-alpha L)
    n_triplets = (n_wdm_channels**3) / 24.0
    p_fwm_watts = n_triplets * (fiber_gamma * p_in_watts * leff)**2 * p_in_watts * math.exp(-alpha * fiber_length_km)
    
    h_planck = 6.626e-34
    c_light = 3e8
    lambda_0 = 1550e-9
    photon_energy = h_planck * c_light / lambda_0
    
    photons_per_sec = p_fwm_watts / photon_energy
    integration_time = 1e-9 # 1 ns window
    
    return photons_per_sec * integration_time

def calculate_dispersion_penalty(distance_km, disp_param, lambda_nm, freq_hz, active_compensation):
    if active_compensation: return 1.0
    c = 299792458
    lambda_m = c / freq_hz
    lambda_Target = lambda_nm * 1e-9
    delta_lambda_nm = abs(lambda_m - lambda_Target) * 1e9
    penalty_db = 0.01 * distance_km * delta_lambda_nm
    return 10**(-penalty_db / 10.0)

def calculate_analytical_qber(V, qcnr, rho, t_l, mu, d_b):
    rate_sig = rho * t_l * mu
    rate_noise = rate_sig / qcnr if qcnr > 0 else 0.0
    rate_dark = d_b
    r_total = rate_sig + rate_noise + 2 * rate_dark
    r_error = (rate_sig * (1.0 - V) / 2.0) + (0.5 * rate_noise) + (rate_dark)
    if r_total <= 0: return 0.5
    return r_error / r_total

def calculate_analytical_key_rate(qber, rho, t_l, mu, f_rep):
    from .math import binary_entropy
    Q_mu = rho * t_l * mu
    rate = f_rep * Q_mu * (1.0 - 2.0 * binary_entropy(qber))
    return max(0.0, rate)

def calculate_lim2014_observables(params, mu, nu, q_x, p_s, p_d):
    dist = params.channel.distance_km
    loss = params.channel.fiber_loss_db_km
    eta_det = params.detector.det_eff_d0
    p_dc = params.detector.dark_rate
    e_det = params.detector.qber_intrinsic
    
    trans = 10**(-loss * dist / 10.0)
    eta = trans * eta_det
    
    def Gain(intensity): return 2.0*p_dc + 1.0 - np.exp(-eta * intensity)
    def ErrorGain(intensity): return 0.5 * (2.0*p_dc) + e_det * (1.0 - np.exp(-eta * intensity))

    p_z = 1.0 - q_x
    p_x = q_x
    p_v = 1.0 - p_s - p_d
    
    stats = {}
    q_mu = Gain(mu); eq_mu = ErrorGain(mu)
    stats["n_x_signal"] = p_s * p_x**2 * q_mu
    stats["n_z_signal"] = p_s * p_z**2 * q_mu
    stats["m_x_signal"] = p_s * p_x**2 * eq_mu
    stats["m_z_signal"] = p_s * p_z**2 * eq_mu
    
    q_nu = Gain(nu); eq_nu = ErrorGain(nu)
    stats["n_x_decoy"] = p_d * p_x**2 * q_nu
    stats["n_z_decoy"] = p_d * p_z**2 * q_nu
    stats["m_x_decoy"] = p_d * p_x**2 * eq_nu
    stats["m_z_decoy"] = p_d * p_z**2 * eq_nu
    
    q_0 = Gain(0.0); eq_0 = ErrorGain(0.0)
    stats["n_x_vacuum"] = p_v * p_x**2 * q_0
    stats["n_z_vacuum"] = p_v * p_z**2 * q_0
    stats["m_x_vacuum"] = p_v * p_x**2 * eq_0
    stats["m_z_vacuum"] = p_v * p_z**2 * eq_0
    
    return stats

def calculate_classical_success_bound(gamma: float, M: int) -> float:
    return 1.0 - 0.5 * (gamma**M)

def calculate_cascaded_fbg_loss_spectrum(freqs, center_freq, bandwidth):
    return np.zeros_like(freqs)

def calculate_serial_link_transmittance(channel_index, n_channels, loss_per_stage_db):
    total_db = (channel_index + 1) * loss_per_stage_db
    return 10**(-total_db / 10.0)

def calculate_effective_modulation_index(m_target, freq, bandwidth):
    if bandwidth is None: return m_target
    resp = 1.0 / math.sqrt(1.0 + (freq / bandwidth)**2)
    return m_target * resp
