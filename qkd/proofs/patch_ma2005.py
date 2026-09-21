#!/usr/bin/env python3
"""
patch_ma2005.py — Apply all bug fixes (B1–B6 + Eq 12 solver) to ma2005.py.

Usage:
    python patch_ma2005.py [path/to/ma2005.py]

Fixes:
  B1: Remove intensity-jitter prefactor in Ma2005VacuumWeakProof (security)
  B2: Implement Eq (45) q = N_S / (2N) instead of hardcoded 0.5
  B3: Remove Chernoff CI override in Ma2005BaseProof.__init__
  B4: Fix eta = t_AB * eta_Bob (Eq 5) via _get_eta() helper
  B5: Fix Y_0 = total background rate (Table 1) via _get_y0() helper
  B6: Document Eq (40) Y_1^{L,mu,0} ambiguity
  +:  Add Eq (12) optimal-mu solver (bisection)
"""

import sys
from pathlib import Path


def tolerant_replace(src: str, old: str, new: str):
    """Replace old→new in src, tolerant of trailing whitespace.

    Matches by stripping trailing whitespace from each line before
    comparison, so '    ' and '' are treated as equal.  Preserves
    the original file's whitespace outside the replaced block.
    """
    old_lines = old.split("\n")
    new_lines = new.split("\n")
    src_lines = src.split("\n")
    old_stripped = [line.rstrip() for line in old_lines]

    for i in range(len(src_lines) - len(old_lines) + 1):
        match = True
        for j in range(len(old_lines)):
            if src_lines[i + j].rstrip() != old_stripped[j]:
                match = False
                break
        if match:
            result = src_lines[:i] + new_lines + src_lines[i + len(old_lines):]
            return "\n".join(result), True
    return src, False


def main():
    filepath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ma2005.py")

    if not filepath.exists():
        print(f"ERROR: {filepath} not found")
        sys.exit(1)

    src = filepath.read_text()
    results = []

    def fix(old, new, label):
        nonlocal src
        src, found = tolerant_replace(src, old, new)
        results.append((label, found))

    # ═══════════════════════════════════════════════════════════════
    # B3: Remove Chernoff CI override
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''    def __init__(self, params: QKDParams, *args, **kwargs):
        # Enforce CHERNOFF CI for finite-key security.
        # The Y1_L formula involves subtraction of exp(mu)-amplified terms,
        # so wide CI bounds (e.g. Gaussian) make Y1_L go negative.
        # Chernoff's multiplicative bounds are tight enough to keep Y1_L > 0.
        if params.ci_method != ConfidenceBoundMethod.CHERNOFF:
            logger.warning(
                f"Ma2005 proof requires CHERNOFF bounds for finite-key security. "
                f"Overriding '{params.ci_method.name}' with CHERNOFF."
            )
            object.__setattr__(params, 'ci_method', ConfidenceBoundMethod.CHERNOFF)

        super().__init__(params, *args, **kwargs)''',
        '''    def __init__(self, params: QKDParams, *args, **kwargs):
        # Spec Sec 4.4: 10-sigma Gaussian ("standard error analysis").
        # Do NOT override user's CI method choice.
        super().__init__(params, *args, **kwargs)''',
        "B3 (Chernoff override removal)"
    )

    # ═══════════════════════════════════════════════════════════════
    # B2 + B4 + B5 + Eq(12): Insert helper methods after _calculate_f_ec
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''    def _calculate_f_ec(self, error_rate: float) -> float:
        """Calculates adaptive error correction efficiency if configured."""
        config = self.p.f_ec_dynamic_config
        if config and config.get("model") == "linear":
            base = config.get("base", 1.1)
            slope = config.get("slope", 0.0)
            return base + slope * error_rate
        return self.p.f_error_correction

    def _calculate_weak_gllp_rate(''',
        '''    def _calculate_f_ec(self, error_rate: float) -> float:
        """Calculates adaptive error correction efficiency if configured."""
        config = self.p.f_ec_dynamic_config
        if config and config.get("model") == "linear":
            base = config.get("base", 1.1)
            slope = config.get("slope", 0.0)
            return base + slope * error_rate
        return self.p.f_error_correction

    def _get_eta(self) -> float:
        """Paper Eq (5): eta = t_AB * eta_Bob.
        eta_Bob = t_Bob * eta_D is given directly in Table 1 (GYS: 0.045,
        KTH: 0.143) as a primitive parameter."""
        eta_bob = getattr(self.p.detector, "eta_bob", None)
        if eta_bob is None:
            t_bob = getattr(self.p.detector, "t_bob", 1.0)
            eta_d = getattr(self.p.detector, "det_eff_d0", 0.0)
            eta_bob = t_bob * eta_d
        return float(self.p.channel.transmittance * eta_bob)

    def _get_y0(self, stats_map=None) -> float:
        """Paper Table 1: Y_0 is the *total* background rate (GYS: 1.7e-6,
        KTH: 4e-4), not per-detector.  Do NOT multiply dark_rate by 2.
        If vacuum stats are available, use Q_vacuum = Y_0 (Eq 33)."""
        if stats_map:
            vac = stats_map.get("vacuum")
            if vac is not None and vac.sent > 0:
                return float(vac.sifted) / float(vac.sent)
        y0 = getattr(self.p.detector, "y0", None)
        if y0 is not None:
            return float(y0)
        return float(getattr(self.p.detector, "dark_rate", 0.0))

    def _calculate_sifting_factor_q(self, signal_sent: int, total_sent: int) -> float:
        """Paper Eq (45): q = N_S / (2N) for standard BB84.
        Asymptotic limit: N_S -> N gives q -> 1/2.
        Set self.p.sifting_factor_q to override (e.g. efficient BB84)."""
        override = getattr(self.p, "sifting_factor_q", None)
        if override is not None:
            return float(override)
        if total_sent > 0:
            return float(signal_sent) / (2.0 * float(total_sent))
        return 0.5

    @staticmethod
    def solve_optimal_mu(e_detector: float, f_ec: float, tol: float = 1e-12) -> float:
        """Paper Eq (12): (1-mu)*exp(-mu) = f(e_det)*H2(e_det)/(1-H2(e_det)).
        Solve for mu in (0, 1] via bisection.
        Reference: GYS f=1.0 -> mu~0.54; GYS f=1.22 -> mu~0.48."""
        h2 = binary_entropy(e_detector)
        if h2 >= 1.0:
            return 1.0
        rhs = f_ec * h2 / (1.0 - h2)

        def g(mu):
            return (1.0 - mu) * math.exp(-mu) - rhs

        if g(1.0) >= 0.0:
            return 1.0
        lo, hi = 1e-12, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            g_mid = g(mid)
            if abs(g_mid) < tol or (hi - lo) < tol:
                return mid
            if g_mid > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _calculate_weak_gllp_rate(''',
        "B2+B4+B5+Eq12 (helper methods)"
    )

    # ═══════════════════════════════════════════════════════════════
    # B2: Fix q in _calculate_weak_gllp_rate
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''        if delta >= 1.0 - 1e-9:
            return 0.0

        q = 0.5
        f_ec = self._calculate_f_ec(E_mu)''',
        '''        if delta >= 1.0 - 1e-9:
            return 0.0

        total_sent = sum(s.sent for s in stats_map.values())
        q = self._calculate_sifting_factor_q(signal_stats.sent, total_sent)
        f_ec = self._calculate_f_ec(E_mu)''',
        "B2 (weak_gllp q)"
    )

    # ═══════════════════════════════════════════════════════════════
    # B2: Fix q in _calculate_gllp_rate
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''        mu = self.p.source.get_pulse_config_by_name("signal").mean_photon_number
        q_factor = 0.5

        Q_mu = signal_stats.sifted / signal_stats.sent''',
        '''        mu = self.p.source.get_pulse_config_by_name("signal").mean_photon_number
        total_sent = sum(s.sent for s in stats_map.values())
        q_factor = self._calculate_sifting_factor_q(signal_stats.sent, total_sent)

        Q_mu = signal_stats.sifted / signal_stats.sent''',
        "B2 (gllp q)"
    )

    # ═══════════════════════════════════════════════════════════════
    # B1: Remove intensity jitter (header block)
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''        mu_base = cfg_sig.mean_photon_number
        nu_base = cfg_decoy.mean_photon_number

        # Input Fluctuation Correction
        # Adjust effective intensities to account for statistical fluctuation in photon number distribution
        # For a finite N, the fraction of n-photon pulses is not exactly P(n|mu).
        # We model this as an uncertainty on mu.
        delta = self.p.source.intensity_jitter

        # Add statistical uncertainty to jitter if N is finite
        n_total = sum(s.sent for s in stats_map.values())
        if n_total > 0:
            # Approx relative fluctuation 1/sqrt(N)
            stat_fluc = 1.0 / math.sqrt(n_total)
            delta += stat_fluc

        mu_min, mu_max = mu_base * (1 - delta), mu_base * (1 + delta)
        nu_min, nu_max = nu_base * (1 - delta), nu_base * (1 + delta)

        alpha = self.eps_alloc.eps_pe''',
        '''        mu = cfg_sig.mean_photon_number
        nu = cfg_decoy.mean_photon_number

        # Spec Sec 4.1: intensity fluctuations explicitly neglected.
        alpha = self.eps_alloc.eps_pe''',
        "B1 (jitter removal - header)"
    )

    # ═══════════════════════════════════════════════════════════════
    # B1: Fix Eq 34 / 36 / 37 terms
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''        # [Paper Eq 34] Lower Bound on Y1
        term_Q_nu = Q_nu_L * math.exp(nu_min)
        term_Q_mu = Q_mu_U * math.exp(mu_max) * (nu_max**2 / mu_min**2)
        term_Y0 = Y0_U * (mu_max**2 - nu_min**2) / (mu_min**2)

        denom = mu_max * nu_min - nu_max**2
        if denom <= 0:
            prefactor = 0.0
        else:
            prefactor = mu_min / denom

        Y1_L = prefactor * (term_Q_nu - term_Q_mu - term_Y0)
        Y1_L = max(0.0, Y1_L)

        # [Paper Eq 37] Upper Bound on e1
        if Y1_L * nu_min <= 1e-20:
            e1_U = 0.5
        else:
            numerator_e1 = (E_nu_U * Q_nu_U * math.exp(nu_max)) - (e0 * Y0_L)
            e1_U = numerator_e1 / (Y1_L * nu_min)
            e1_U = min(0.5, max(0.0, e1_U))

        # [Paper Eq 36] Tagged Fraction Delta Upper Bound
        if Q_nu_L > 0:
            term1 = (nu_max / (mu_min - nu_max))
            term2 = (nu_max * math.exp(-nu_min) * Q_mu_U) / (mu_min * math.exp(-mu_max) * Q_nu_L) - 1.0
            term3 = (nu_max * math.exp(-nu_min) * Y0_U) / (mu_min * Q_nu_L)
            delta_bound_strict = term1 * term2 + term3
        else:
            delta_bound_strict = 1.0''',
        '''        # [Paper Eq 34] Lower Bound on Y1
        term_Q_nu = Q_nu_L * math.exp(nu)
        term_Q_mu = Q_mu_U * math.exp(mu) * (nu**2 / mu**2)
        term_Y0 = Y0_U * (mu**2 - nu**2) / (mu**2)

        denom = mu * nu - nu**2
        if denom <= 0:
            prefactor = 0.0
        else:
            prefactor = mu / denom

        Y1_L = prefactor * (term_Q_nu - term_Q_mu - term_Y0)
        Y1_L = max(0.0, Y1_L)

        # [Paper Eq 37] Upper Bound on e1
        if Y1_L * nu <= 1e-20:
            e1_U = 0.5
        else:
            numerator_e1 = (E_nu_U * Q_nu_U * math.exp(nu)) - (e0 * Y0_L)
            e1_U = numerator_e1 / (Y1_L * nu)
            e1_U = min(0.5, max(0.0, e1_U))

        # [Paper Eq 36] Tagged Fraction Delta Upper Bound
        if Q_nu_L > 0:
            term1 = (nu / (mu - nu))
            term2 = (nu * math.exp(-nu) * Q_mu_U) / (mu * math.exp(-mu) * Q_nu_L) - 1.0
            term3 = (nu * math.exp(-nu) * Y0_U) / (mu * Q_nu_L)
            delta_bound_strict = term1 * term2 + term3
        else:
            delta_bound_strict = 1.0''',
        "B1 (Eq 34/36/37 terms)"
    )

    # ═══════════════════════════════════════════════════════════════
    # B4 + B5: Fix eta and Y0 in Ma2005VacuumWeakProof asymptotic ref
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''        # Asymptotic Reference
        eta = self.p.channel.transmittance * self.p.detector.det_eff_d0
        y0_val = 2.0 * self.p.detector.dark_rate   # Gap [B] fix: two-detector BB84
        e_det = self.p.detector.qber_intrinsic''',
        '''        # Asymptotic Reference (B4+B5: use _get_eta / _get_y0 helpers)
        eta = self._get_eta()
        y0_val = self._get_y0(stats_map)
        e_det = self.p.detector.qber_intrinsic''',
        "B4+B5 (VW asymptotic ref)"
    )

    # ═══════════════════════════════════════════════════════════════
    # B6: Document Eq 40 ambiguity in Ma2005OneDecoyProof
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''            if Y1_L * mu <= 1e-20: e1_U = 0.5
            else: e1_U = min(0.5, max(0.0, (E_mu_U * Q_mu_U * math.exp(mu)) / (Y1_L * mu)))
            diag_msg = "Ma2005_OneDecoy_Simple"''',
        '''            if Y1_L * mu <= 1e-20:
                e1_U = 0.5
            else:
                # NOTE (B6): Spec Eq (40) uses Y_1^{L,mu,0} which is not
                # explicitly defined in the paper (spec Sec 12 #3). We use
                # Y1_L from Eq (39) as a conservative substitute.
                e1_U = min(0.5, max(0.0, (E_mu_U * Q_mu_U * math.exp(mu)) / (Y1_L * mu)))
            diag_msg = "Ma2005_OneDecoy_Simple"''',
        "B6 (Eq 40 ambiguity doc)"
    )

    # ═══════════════════════════════════════════════════════════════
    # B5: Fix Y0 in Ma2005AsymptoticProof
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''        Y0 = 2.0 * self.p.detector.dark_rate   # Gap [B] fix: two-detector BB84
        e0 = 0.5
        e_det = self.p.detector.qber_intrinsic''',
        '''        # Paper Table 1: Y_0 is the total background rate.
        Y0 = self._get_y0(stats_map)
        e0 = 0.5
        e_det = self.p.detector.qber_intrinsic''',
        "B5 (AsymptoticProof Y0)"
    )

    # ═══════════════════════════════════════════════════════════════
    # B4: Fix eta fallback in Ma2005AsymptoticProof
    # ═══════════════════════════════════════════════════════════════
    fix(
        '''        if arg <= 0:
            # Q_mu is anomalously high (physical impossibility or noise), implies eta is undefined.
            # Fallback: use theoretical eta from config if available for stability
            eta = self.p.channel.transmittance * self.p.detector.det_eff_d0
        else:
            eta_effective = -math.log(arg) / mu
            # Clamp to physical reality
            eta = max(0.0, min(1.0, eta_effective))''',
        '''        if arg <= 0:
            # Q_mu anomalously high; fall back to channel-model eta.
            eta = self._get_eta()
        else:
            eta_effective = -math.log(arg) / mu
            eta = max(0.0, min(1.0, eta_effective))''',
        "B4 (AsymptoticProof eta fallback)"
    )

    # ── Write patched file ──
    filepath.write_text(src)

    # ── Report ──
    ok_count = sum(1 for _, f in results if f)
    skip_count = sum(1 for _, f in results if not f)
    print(f"\nPatched {filepath}")
    print(f"  {ok_count} applied, {skip_count} skipped\n")
    for label, found in results:
        status = "OK  " if found else "SKIP"
        print(f"  [{status}] {label}")
    print()


if __name__ == "__main__":
    main()
