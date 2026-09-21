# qkd/detectors.py
# -*- coding: utf-8 -*-
"""
Physically realistic models for single-photon detectors.

This module implements an event-driven single-photon detector model for a
decoy-state BB84 quantum key distribution (QKD) simulator.  It supports
SPAD (threshold), SNSPD, and PNRD (photon-number-resolving) detector
types, with dead time, afterpulsing, dark counts, timing jitter,
chromatic-dispersion broadening, and an optional gated-detector mode.

Features
--------
- Event-driven threshold-detector simulation for SPAD-like detectors with
  dead time, afterpulsing, dark counts, timing jitter, and chromatic
  dispersion broadening.
- A separate SNSPD-tuned parameter interpretation
  (``DetectorType.SNSPD``) that uses SNSPD-appropriate dark-rate scaling
  (Planck-spectrum black-body model plus bias-current scaling) and skips
  the SPAD-specific Geiger-mode guard.
- PNRD simulation path with count-valued outputs using proper Binomial
  thinning (no double-Poisson, no cross-detector double counting).
- Asymmetric detector efficiencies, asymmetric dark count rates, and
  asymmetric per-detector timing jitter.
- Stateful batch simulation with serializable detector runtime state.
- Optional gated-detector mode (dark counts and signal are only
  registered inside narrow gate windows synchronized to the pulse grid).
- Entangling decoder helper with CPTP-style visibility-aware dephasing.

Design notes / known simplifications
------------------------------------
- The PNRD path is intentionally **stateless** w.r.t. dead time,
  afterpulse and continuous-time dark-count scheduling.  Per-pulse
  Poisson dark counts are used for PNRD; this is documented and
  consistent within the path.  ``strict_mode`` (now the default) enforces
  that PNRD is not combined with non-zero ``dead_time_ns`` /
  ``afterpulse_prob`` / ``jitter_fwhm_ns``.  The PNRD path DOES update
  ``total_time_processed_ns`` so that switching from PNRD to threshold
  across batches preserves absolute-time consistency.

  Review-v8 R9-01 (HIGH): the PNRD path also uses per-PULSE routing
  (all photons in a multi-photon pulse go to the SAME detector),
  while the threshold path defaults to per-PHOTON routing.  This
  silently biases decoy-state yields ``Y_n`` for ``n >= 2``.
  ``strict_mode`` now raises when the PNRD path receives multi-photon
  input; callers needing per-photon routing for decoy-state yield
  analysis should use the threshold path (SPD/SNSPD).
- The SNSPD model reuses the event-driven engine but applies a distinct
  dark-rate temperature scaling based on a Planck-spectrum black-body
  photon-flux integral (replacing the previous Arrhenius approximation)
  and a bias-current scaling factor.  The SPAD Geiger-mode check is
  skipped for SNSPD/PNRD.

  Review-v8 R9-60 (PHYSICS CAVEAT): the SNSPD dark-rate scalers
  (Planck-spectrum black-body + empirical bias-current factor) are
  EXPLORATORY / QUALITATIVE ONLY.  They are NOT calibrated against
  characterization data and are not suitable for publication-bound
  work without explicit caveats.  For publication-bound SNSPD work,
  callers MUST supply measured dark rates at the operating point
  (set ``dark_rate`` to the measured value and use
  ``ref_temperature_k = temperature_k`` /
  ``ref_bias_voltage = bias_voltage`` so the scaler evaluates to 1.0).
  The strict-mode 5% scaler-deviation guard (review-v7 R7-04/R7-05)
  enforces this by raising when the scaler deviates from 1.0 by more
  than 5%.
- Detector efficiency during dead-time recovery is modeled as a hard
  cut-off (no gradual ramp); this is a known simplification.
- Afterpulse cascade depth is bounded by ``MAX_AFTERPULSE_CASCADE`` to
  prevent runaway chains.  The bound is enforced **before** scheduling a
  new afterpulse, not after.
- **Neither the PNRD nor the threshold path models detector bandwidth
  or maximum count rate** (review-v8 R9-40/R9-41).  Real detectors
  have a finite electronic bandwidth (typically ~10 MHz to ~1 GHz for
  SPADs, ~GHz for SNSPDs) that limits the maximum sustainable count
  rate; above the bandwidth limit, pulses pile up and the detector's
  effective efficiency drops.  The non-paralyzable dead-time model
  handles this partially (dead time suppresses subsequent clicks),
  but the model does not account for the detector's finite electronic
  bandwidth or for pulse-shaping effects.  At very high photon rates
  (approaching ``1 / dead_time_ns``), the model's predictions become
  increasingly non-physical.  Callers should validate that the
  simulated photon rate is well below the detector's bandwidth limit;
  a future revision may add a ``max_count_rate`` parameter.
- Timing jitter is applied **uniformly at the binning stage** to every
  fire event (signal, IMD, dark count, afterpulse).  The kernel
  processes "true" event times; dead-time, afterpulse, and dark-count
  scheduling all use true times.  This is the physically correct model:
  the dead-time clock starts at the true photon arrival time, while the
  recorded (jittered) time is what gets binned into pulse slots.
- **Chromatic-dispersion broadening** is applied **only to
  signal-originated fires** (review-v12 F3).  Dark counts and
  afterpulses originate inside the detector and do not traverse the
  fiber; they should not experience chromatic dispersion.  The previous
  code applied dispersion to ALL fires, inflating ISI and gate-loss
  statistics for noise-originated events.
- Dark-count inter-arrival times use a low-side clamp of
  ``log_safe_eps = 1e-300`` (the smallest positive double is ~5e-324).
  This gives a maximum inter-arrival wait of ``~690/rate`` -- effectively
  unbounded for any realistic batch duration.  The previous high-side
  clamp (``min(1-eps, u)``) was removed because it biased the
  exponential/geometric distribution toward short delays and truncated
  the long-wait tail of the dark-count Poisson process.  ``u == 1.0``
  yields ``-log(1) = 0`` (zero-wait event), which is physically valid;
  ``u == 0.0`` is guarded by the low-side clamp.

  Review-v6 F26 fix: the previous module docstring claimed the
  inter-arrival was clamped to ``[EPS, 1-EPS]`` with ``EPS = 1e-15``,
  which is stale -- the kernel has used ``log_safe_eps = 1e-300``
  (low-side only) since review-v4.  The docstring now matches the
  implementation.

Review-v6 audit
---------------
This module was audited against ``detectors_review_6.tex`` (Ruthless
Technical Review, 2026-07-27).  All Critical (C1-C7) and High-severity
issues identified in that review have been addressed; the per-fix
rationale is documented in inline ``Review-v6`` comments and the
accompanying ``CHANGELOG.md``.  See the review document for the
authoritative list of findings; the inline comments provide the
mapping back to specific finding IDs (F1-F65).

Review-v12 audit
----------------
This module was audited against ``detectors_review_12.tex`` (Ruthless
Technical Review, 2026-07-28).  Critical fixes implemented:

- **F1 (Critical):** RNG state restoration after kernel pre-draw was
  removed.  The previous code restored the RNG state after
  ``rng.random(est_events)`` so that post-kernel jitter and
  double-click resolution would start from the pre-draw state.  This
  deterministically correlated kernel and post-kernel RNG draws,
  biasing ISI, gate-loss, and double-click statistics.  The fix: let
  post-kernel draws naturally follow the pre-draw.

- **F2 (Critical):** UNIFORM AP rescheduling was fixed to release
  AFTER the dead-time window, not inside it.  The previous code used
  ``release_time = last_fire + u_release * dt_ns``, which could cause
  infinite rescheduling loops when the AP delay was small.  The fix:
  ``release_time = last_fire + dt_ns + u_release * ap_lifetime``.

- **F3 (High):** Dispersion broadening is now applied only to
  signal-originated fires.  Dark counts and afterpulses do not traverse
  the fiber and should not experience chromatic dispersion.  The kernel
  now tracks ``is_sig`` per fire and the binning stage uses it to
  select per-fire sigma.

- **F5 (High):** ``tossed_breakdown`` no longer subtracts
  ``deadtime_dropped`` (which was never an addend of ``tossed_events``).

Review-v13 audit
----------------
This module was audited against ``detectors_review_13.tex`` (Ruthless
Technical Review, 2026-07-28).  Critical fixes implemented:

- **F1 (Critical):** AP rescheduling double-delay bug fixed.  The
  previous reschedule code drew a fresh AP delay from the afterpulse
  distribution and added it on top of the release time, double-delaying
  the AP and biasing AP timing statistics (mean delay approximately
  doubled, variance inflated).  The fix: when rescheduling an AP
  during dead time, the AP fires at the release time (plus the minimal
  ``AP_RESCHEDULE_SLACK_NS`` for edge-case safety) without an
  additional delay draw.  The original delay semantics were already
  consumed when the AP was first scheduled.

- **F7 (High):** Per-pulse ``flips_mask`` RNG draw is now deferred
  until the routing mode is determined.  The previous code drew
  ``rng.random(num_pulses) < p_combined_flip`` unconditionally, even
  when per-photon routing was active and a separate per-photon flip
  mask would be drawn later.  This wasted RNG draws and broke RNG
  sequence correspondence between per-pulse and per-photon modes
  (different RNG consumption order produced different results for the
  same seed).  The fix: draw ``flips_mask`` only in the per-pulse
  routing branch.

Review-v15 audit
----------------
This module was audited against ``detectors_review_15.tex`` (Ruthless
Technical Review, 2026-07-28).  Critical and high-severity fixes
implemented:

- **F-01 (Critical):** ``tossed_events`` / ``tossed_breakdown``
  inconsistency fixed.  ``tossed_events`` now includes
  ``double_clicks_discarded`` and ``rogers_discarded_events``
  (added in the policy-specific ``replace()`` calls); the
  ``tossed_breakdown`` decomposition is now exact without the
  ``max(0, ...)`` clamp that previously masked kernel-tossed
  dark/AP events.

- **F-02 (Critical):** RNG retry reproducibility guarded.  When a
  retry changes the RNG buffer size, the kernel consumes a different
  random sequence.  In strict mode, retries that change the buffer
  size now raise ``ParameterValidationError`` instead of silently
  proceeding with a different sequence.

- **F-03 (Critical):** AP overwrite tracking.  When a new AP
  overwrites a pending AP on the same detector, the overwritten AP
  is now counted in ``DIAG_TOSSED_CORE`` so it appears in
  diagnostics.  The underlying single-pending-AP limitation is
  documented and guarded by the occupancy check.

- **F-04 (Critical):** Long-run float64 guard now checks
  ``total_time_processed_ns`` after each batch as well as before.
  A single long batch can blow past the pre-batch limit.

- **F-05 (High):** IMD post-jitter approximation no longer re-jitters
  IMD-only fires.  The previous code drew a separate ``rng.normal``
  for IMD-only events, double-counting jitter RNG consumption and
  biasing the diagnostic.  ``imd_absorbed_by_signal_post_jitter`` is
  now set equal to ``imd_absorbed_by_signal_pre_jitter`` in all
  cases; they are exactly equal when jitter is zero.

- **F-11 (High):** Rogers sifting jitter/dead-time ratio guard added.
  When ``jitter_fwhm_ns > 0.5 * dead_time_ns``, the physical
  assumption that jitter does not move events across sequence
  boundaries is violated.  Strict mode now raises.

- **F-17 (Medium):** Kernel iteration cap now scales with batch size.
  ``max_iterations = min(max(10M, 100 * est_total_events), 100M)``.

- **F-19 (Medium):** ``ap_depth`` is now reset to 0 on ANY fire
  (AP or non-AP) that does not trigger a new AP.  The previous code
  only reset depth on AP fires, leaving depth at the cascade limit
  for non-AP fires and suppressing subsequent legitimate APs.

- **F-20 (Medium):** Preserved AP/DC times past batch end are now
  clamped to ``time_end_abs`` instead of ``current_time``.  This
  prevents them from being silently lost on the next batch as
  ``pre_batch_lost`` events.

- **F-21 (Medium):** Dark-count D0/D1 tie-breaking now uses ``<=``
  (consistent with input/AP priority), not ``<``.

- **F-30 (Medium):** Float-to-bool conversion in const_params uses
  ``>= 0.5`` instead of ``> 0.5``, giving True for the exact 0.5
  midpoint.

- **F-35 (Medium):** SNSPD Planck series convergence check changed
  from ``e_neg_x < 0.999`` (which skipped iteration for x near 0,
  under-estimating flux by ~20%) to ``e_neg_x > 1e-300`` (iterate
  up to MAX_TERMS, relying on relative-tolerance break).

- **F-38 (Low):** Stray comment fragment ``"separately above)."``
  removed.

- **F-81 (Low):** Unknown double-click policies are no longer
  silently treated as KEEP_BOTH.  The ``else`` branch is now an
  explicit ``elif`` for KEEP_BOTH; unknown policies raise
  ``ConfigurationError``.

Known scientific limitations (review-v13)
------------------------------------------
- **F2:** RNG buffer size is a behavioral parameter.  Different
  ``rng_buffer_mult`` values produce different random sequences for the
  same seed because the kernel consumes floats from a pre-drawn array
  whose length depends on ``rng_buffer_mult``.  Strict mode (the
  default) rejects non-default buffer sizes unless
  ``allow_nondefault_buffer=True`` is explicitly set.  For
  publication-bound work, fix ``rng_buffer_mult`` at the default value
  (10.0) and document it as a fixed parameter.
- **F3:** The XOR-combined flip probability formula
  ``m*(1-q) + (1-m)*q`` assumes statistical independence between
  misalignment and QBER/visibility errors, which is physically wrong
  for BB84.  Strict mode already requires ``flip_prob_override`` or
  explicit ``allow_xor_flip_formula=True``.
- **F4:** ``total_time_processed_ns`` is float64.  At 10^14 ns (~11.5
  days at 1 GHz) the ULP is ~22 ns, comparable to typical dead times.
  The existing long-run guard raises in strict mode at batch start
  AND after each batch (review-v15 F-04); callers should split into
  shorter batches.
- **F6:** When the AP cascade depth limit is hit, ``ap_depth`` is
  set to the limit value (not reset to 0), preventing a new cascade
  from starting immediately on the next AP-originated fire.  A new
  cascade can only start from a non-AP fire (signal/dark/IMD), which
  resets depth to 1.  When an AP-originated fire does NOT trigger a
  new AP, the cascade ends and depth is reset to 0.  When a non-AP
  fire does NOT trigger a new AP, depth is reset to 0 ONLY if there
  is no pending AP on that detector (review-v16 F-C1 fix); the
  pending AP's depth is independent of the non-AP fire.  The
  strict-mode occupancy guard (10%) mitigates the most dangerous
  regime but is a heuristic.
- **F8:** Rogers 2007 sequence detection uses true times
  (``final_click_times``) for the gap comparison against dead time,
  while binning uses jittered times.  This is physically correct: the
  detector's dead-time clock starts at the true photon arrival time,
  not the jittered measurement time.  The review's suggestion to use
  jittered times for sequence detection would be incorrect.
- **F9/F10:** PNRD path uses per-pulse Poisson dark counts (no
  continuous-time scheduling) and does not support basis-dependent
  efficiency.  Both gaps are guarded in strict mode.
- **F11:** ``imd_absorbed_by_signal_post_jitter`` is set equal to
  ``imd_absorbed_by_signal_pre_jitter`` in all cases (review-v15
  F-05).  The previous code attempted a separate re-jitter of IMD-only
  fires, which double-counted jitter RNG consumption and biased the
  diagnostic.  When jitter is zero, both values are exactly equal.
  When jitter is nonzero, the post-jitter value is an approximation
  that may be slightly inaccurate; the fully-correct value requires
  tracking IMD-origin flags through the kernel output buffer (future
  work).

- **F8 (Medium):** Per-detector Geiger-mode checks added for
  ``bias_voltage_d0`` / ``bias_voltage_d1`` overrides.

- **F13 (Medium):** No-more-events check moved above RNG/output
  exhaustion checks in the kernel loop, preventing spurious exhaustion
  when the loop would terminate cleanly.

Other findings are documented as limitations or deferred; see the
inline ``Review-v12`` comments for the full mapping.

Known scientific limitations (review-v12)
------------------------------------------
- **F4:** The default combined-flip probability uses the XOR formula
  ``m*(1-q) + (1-m)*q``, which assumes statistical independence
  between misalignment and QBER/visibility errors.  In BB84, these
  are correlated through the same physical imperfections (visibility
  degradation).  The XOR formula can overestimate the combined error
  by up to ~2x in the small-error limit.  Strict mode (the default)
  now REQUIRES ``flip_prob_override`` or explicit
  ``allow_xor_flip_formula=True``.  For publication-bound work,
  always supply ``flip_prob_override`` computed from a physical model.
- **F6:** Rogers 2007 simultaneous D0&D1 clicks are currently resolved
  by keeping D0 (convention).  This biases the sifted bit toward D0.
  Callers needing arrival-order resolution should implement it in the
  post-sifting step using the jittered times.
- **F9/F10:** The single-pending-AP approximation and cascade-depth
  reset allow new AP cascades to start immediately after the limit is
  hit.  This underestimates long-tail AP contribution for high
  ``afterpulse_prob`` and long ``afterpulse_lifetime_ns``.  The
  strict-mode occupancy guard (10%) mitigates the most dangerous
  regime but is a heuristic, not a proof.
- **F14:** ``total_time_processed_ns`` is float64.  At 10^14 ns the
  ULP is ~22 ns, comparable to typical dead times.  The long-run
  guard raises in strict mode; callers should split into shorter
  batches.

Review-v16 fixes
----------------
- **F-C1 (Critical):** AP cascade-depth tracking no longer clobbers the
  pending AP's depth when a non-AP fire (signal/dark/IMD) does not
  trigger a new AP.  The previous code unconditionally reset
  ``ap_depth`` to 0 on any fire that didn't trigger an AP, which
  caused the pending AP to fire at depth 1 regardless of its original
  cascade depth, defeating the cascade limit and over-counting
  afterpulses.
- **F-H9 (High):** Stale dark-count times are now invalidated when the
  dark rate changes to zero between batches.  The previous code
  allowed a pending dark count from a previous batch to fire in a
  batch where the dark rate is zero, which is non-physical.
- **F-M4 (Medium):** Kernel sortedness check: ``np.diff`` is not
  Numba-compatible on unaligned structured-array field views.  Kept
  as a Python for-loop with early exit; added a comment documenting
  why.  The loop runs once per kernel invocation (not per event).
- **F-M7 (Medium):** Kernel RNG exhaustion check uses the exact needed
  count (1 when ``ap_prob == 0``, 2 otherwise) instead of always
  requiring 2, avoiding premature exhaustion in AP-free configurations.
- **F-M3 (Medium):** Removed dead ``target_d0`` computation in the PNRD
  per-photon routing branch (was computed but never used).
- **F-M10 (Medium):** PNRD path now emits a warning in non-strict mode
  when ``dead_time_ns``, ``afterpulse_prob``, or ``jitter_fwhm_ns`` is
  non-zero, instead of silently ignoring them.
- **F-L1 (Low):** ``_config_hash()`` now returns the full 64-character
  SHA-256 hex digest instead of a 16-character truncation.  Legacy
  16-char hashes are accepted in ``set_state`` for backward compat.
- **F-L4 (Low):** ``rogers_T_N`` docstring corrected: the geometric
  form is used for all ``N >= 1``, not just ``N >= 5``.

Review-v18 fixes
----------------
- **F-01 (Critical):** IMD double-counting bug fixed.  The event arrays
  ``ev_d0``/``ev_d1`` previously used ``raw_clicks_d0``/``raw_clicks_d1``
  (which includes IMD-only events via ``detected_d0_mask |
  imd_clicks_d0_mask``).  Combined with ``imd_ev_d0``/``imd_ev_d1``
  (which also includes IMD-only events via ``imd_clicks_d0_mask &
  ~detected_d0_mask``), this caused IMD-only events to be
  double-entered into the kernel.  When ``dead_time_ns=0``, both
  duplicates fire, inflating click rates by 1 per IMD-only event.
  When ``dead_time_ns>0``, the second duplicate is dead-time suppressed,
  inflating ``DIAG_DEADTIME``.  The fix: ``ev_d0``/``ev_d1`` now use
  ``detected_d0_mask``/``detected_d1_mask`` (signal-only), leaving
  IMD-only events exclusively in ``imd_ev_d0``/``imd_ev_d1``.
  This bug was introduced by the Review-v12 F3 fix which added the
  separate IMD event arrays for per-fire sigma selection.
- **F-02 (High):** Documented the breakdown-voltage temperature
  dependence assumption in ``_spad_dark_scaler``.  The ``ref_excess_bias``
  computation uses ``self.breakdown_voltage`` for BOTH the operating and
  reference points.  In reality, InGaAs SPAD breakdown voltage shifts
  with temperature at ~50 mV/K.  When ``ref_temperature_k !=
  temperature_k``, the excess-bias calculation is silently wrong.
  For publication-bound work, callers should set
  ``ref_temperature_k = temperature_k`` so the scaler evaluates to 1.0.
- **F-03 (High):** ``_calculate_dynamic_dark_rates`` is now cached per
  ``simulate_detection`` call.  The cache is invalidated at the start
  of each call.  Previously, the method was called at least twice per
  batch (once in ``_validate_simulation_inputs`` and once in the
  simulation path), and for SNSPD each call recomputes the Planck
  integral (up to 200 series terms).

Review-v19 fixes
----------------
- **F-01 (Critical):** Temperature-dependent breakdown voltage correction
  implemented in ``_spad_dark_scaler``.  New parameter
  ``breakdown_temp_coeff_mv_per_k`` (default 50 mV/K for InGaAs)
  adjusts the operating-point breakdown voltage:
  ``bdv_op = breakdown_voltage + coeff * 1e-3 * (temperature_k -
  ref_temperature_k)``.  The previous code used the same
  ``breakdown_voltage`` for both operating and reference excess-bias,
  silently producing wrong dark-rate scaler values when
  ``ref_temperature_k != temperature_k``.  For a 50 K temperature
  difference, the old code computed voltage_factor = 1.0 (both
  excess-bias = 5V), while the corrected code computes
  voltage_factor = 0.25 (excess_bias = 2.5V at 350K vs 5V at 300K).
  Set ``breakdown_temp_coeff_mv_per_k = 0.0`` to revert to the old
  behavior.
- **F-11 (High):** ``DetectionDiagnostics.kernel_tossed`` now populated
  directly from ``DIAG_TOSSED_CORE`` in the kernel diagnostic, instead
  of computed by subtraction in ``tossed_breakdown``.  The previous
  subtraction-based computation could go negative if any component was
  over-counted, masking bugs in diagnostic accounting.  The ``tossed_breakdown``
  property now uses the direct field.
- **F-14 (High):** ``json.dumps`` in ``get_state`` now uses
  ``allow_nan=False``.  If NaN values leak into the detector state
  (which should never happen), the serialization will raise
  ``ValueError`` instead of silently corrupting the JSON output.

Remaining v19 findings (F-02 through F-10, F-12 through F-18) are
modeling risks, architectural limitations, or performance concerns
that the review recommends deferring to future revisions.  They are
documented and guarded in strict mode where applicable:

- F-02: Single-pending-AP approximation (architectural, deferred to v20+)
- F-03: XOR flip formula (guarded in strict mode, requires flip_prob_override)
- F-04: rng_buffer_mult behavioral parameter (guarded, documented)
- F-05: Hard-cutoff dead-time recovery (NotImplementedError in strict mode)
- F-06: Kernel continue-branch flow (maintainability, no bug)
- F-07: Low-side clamp bias (1e-300, negligible)
- F-08: SPAD quadratic excess-bias (known modeling risk, documented)
- F-09: SNSPD qualitative scaler (guarded, documented)
- F-10: PNRD dark-count model (guarded, documented)
- F-12: p_qber additive model (modeling risk)
- F-13: Kernel decomposition (performance, deferred)
- F-15: Cascade depth reset logic (correct but fragile)
- F-16: max_count_rate_hz not modeled (deferred to v20+)
- F-17: Rogers D0 bias (documented convention)
- F-18: Geometric AP tail truncation (guarded by strict mode)

Faithfulness to Rogers et al. (2007), "Detector dead-time effects and
paralyzability in high-speed QKD"
---------------------------------------------------------------------
The dead-time / sifting logic in this module follows the physics and the
sifting algorithm of Rogers, Bienfang, Nakassis, Xu & Clark (2007).
The relevant paper-level invariants enforced here are:

1. **Individual SPADs are non-paralyzable.**  Rogers et al. §3 and §6
   state explicitly that, taken individually, SPADs are non-paralyzable
   counters; photons arriving during dead time have no significant effect
   on the detector.  Paralyzability is an *emergent* property of the
   **basis-level pair of detectors** together with the sifting rule, not
   of a single SPAD.  Accordingly, ``DeadTimeModel.PARALYZABLE`` is
   **deprecated** at the per-detector level: it is accepted for backward
   compatibility but raises in ``strict_mode`` and is otherwise treated
   as non-paralyzable with a warning.  True Rogers-style paralyzability
   is recovered only when the ``ROGERS_2007`` sifting policy is active.

2. **At most one sifted bit per detection sequence.**  When the
   ``ROGERS_2007`` double-click / sifting policy is active, a "detection
   sequence" is defined as the maximal run of detection events on D0
   and D1 such that consecutive events are separated by less than the
   dead time ``τ``.  Such a sequence contributes **at most one** sifted
   bit — the first event in the sequence — and subsequent events in the
   same sequence are discarded (counted in
   ``DetectionDiagnostics.rogers_discarded_events``).

3. **Analytic helpers (Eqs. 8, 9, 10–13, 15, 16, 17).**  The module
   provides ``rogers_P_00``, ``rogers_T_N``, ``rogers_S``,
   ``rogers_sifted_bit_rate``, ``rogers_sbr_max``, and
   ``rogers_rho_tx_max`` implementing the closed-form state-space model
   from Rogers et al. §3–4.  ``rogers_T_N`` implements the standard
   geometric form ``T_N = (1 - s)^(N-1) * s`` with
   ``s = (1 - 2p)^k`` (probability of no follow-up click within the
   dead-time window of ``k`` cycles), which sums to 1 over ``N >= 1``.

Reproducibility
---------------
- ``strict_mode`` defaults to ``True``.  Set to ``False`` only for
  exploratory use where silent degradation is acceptable.
- RNG state is snapshotted before the first simulation attempt and
  restored before each retry, so the same seed always produces the same
  result regardless of whether a buffer-exhaustion retry occurred.
- Numba compilation uses ``fastmath=False`` to guarantee bit-identical
  results between the Numba-compiled and pure-Python kernel paths.
- The state schema is versioned (``STATE_VERSION``); loading a state
  produced by an incompatible code version raises.
- Review-v7 R7-01: ``rng_buffer_mult`` is a BEHAVIORAL parameter, not
  a transparent buffer-size hint.  The kernel consumes floats from a
  pre-drawn array whose length depends on ``rng_buffer_mult``; different
  buffer sizes produce different random sequences for the same seed.
  In strict mode (the default) we raise if the caller supplies a
  non-default ``rng_buffer_mult`` unless they explicitly opt in via
  ``allow_nondefault_buffer=True``.  For publication-bound work, fix
  ``rng_buffer_mult`` at the default value (10.0) and document it as a
  fixed parameter.
- Review-v7 R7-02: ``_RNGStateGuard`` is DEPRECATED.  Its unconditional
  restore-on-exit semantics caused the C1/F1 bug in review-v6.  Use
  ``_copy_rng_state(rng.bit_generator.state)`` with manual re-application
  instead.  Target removal: v19.
- Review-v7 R7-19: ``DetectionDiagnostics.misalignment_flips`` is
  DEPRECATED.  It is now a ``@property`` that emits a
  ``DeprecationWarning`` on access and returns
  ``combined_flips_with_arrivals``.  Target removal: v19.

Public API stability
--------------------
- ``SinglePhotonDetector`` is intentionally stateful and therefore NOT
  frozen.
- ``to_config_dict()`` serializes **configuration** only.
- ``get_state()`` / ``set_state()`` serialize and restore **runtime
  state** only.  ``STATE_VERSION`` is bumped whenever the state schema
  changes; ``set_state`` performs forward migration from a finite set of
  legacy versions.
"""

import copy
import hashlib
import json
import logging
import math
import warnings
from dataclasses import dataclass, field, replace, fields as dataclass_fields
from enum import Enum
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# --- Numba Import Handling ---------------------------------------------------
# We import ``njit`` lazily and never declare an explicit signature on the
# hot-loop kernel.  This avoids brittle ABI dependencies on
# ``numba.types.Tuple`` / ``numba.types.Array`` which have shifted across
# Numba releases.  If compilation fails for any reason, we fall back to the
# pure-Python implementation and emit a single warning.
#
# Compilation uses ``fastmath=False`` to guarantee bit-identical results
# between the Numba-compiled and pure-Python kernel paths.  This is a
# reproducibility requirement (review F20).
NUMBA_AVAILABLE = False
try:
    from numba import njit  # type: ignore
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when numba is absent
    def njit(*args, **kwargs):  # type: ignore
        """Fallback no-op decorator used when Numba is unavailable."""
        def decorator(func):
            return func
        # Support both ``@njit`` and ``@njit(...args, kwargs)`` usage.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

# === QKD package imports (with standalone fallbacks) ========================
try:
    from .datatypes import (  # type: ignore
        DoubleClickPolicy,
        DetectorType,
        DetectionConfig,
        DecoderArchitecture,
        TallyCounts,
    )
    from .exceptions import ConfigurationError, ParameterValidationError  # type: ignore
    from .constants import (  # type: ignore
        is_valid_probability,
        is_finite_non_negative,
        clamp_probability,
        clamp_probabilities,
        is_close,
        EPS,
        CONST_BOLTZMANN,
        CONST_ELECTRON_CHARGE,
        CONST_PLANCK,
        CONST_SPEED_OF_LIGHT,
    )
    from .utils.utils import sanitize_for_serialization  # type: ignore
    from .utils.scm_analysis import (  # type: ignore
        calculate_visibility,
        calculate_intensity_mismatch_factor,
        calculate_wdm_fwm_noise,
    )
    _HAS_QKD_PACKAGE = True
except ImportError:  # pragma: no cover - standalone fallback for testing
    _HAS_QKD_PACKAGE = False

    class DoubleClickPolicy(Enum):
        DISCARD = "discard"
        RANDOM = "random"
        KEEP_BOTH = "keep_both"
        ROGERS_2007 = "rogers_2007"

    class DetectorType(Enum):
        SPD = "spd"
        SNSPD = "snspd"
        PNRD = "pnrd"

    class DecoderArchitecture(Enum):
        ENTANGLING = "entangling"
        DIRECT = "direct"

    @dataclass(frozen=True)
    class DetectionConfig:
        efficiency: float = 1.0
        dark_count_rate: float = 0.0
        detector_type: DetectorType = DetectorType.SPD

    @dataclass(frozen=True)
    class TallyCounts:
        double_clicks_discarded: int = 0

    class ConfigurationError(Exception):
        pass

    class ParameterValidationError(Exception):
        def __init__(self, msg, *, param_name=None, param_value=None):
            super().__init__(msg)
            self.param_name = param_name
            self.param_value = param_value

    EPS = 1e-15
    CONST_BOLTZMANN = 1.380649e-23  # J/K
    CONST_ELECTRON_CHARGE = 1.602176634e-19  # C
    CONST_PLANCK = 6.62607015e-34  # J*s
    CONST_SPEED_OF_LIGHT = 2.99792458e8  # m/s

    def is_valid_probability(p: float) -> bool:
        try:
            # Review-v6: handle numpy arrays gracefully (extract scalar
            # from 0-d / 1-element arrays; reject larger arrays).
            if isinstance(p, np.ndarray):
                if p.shape == () or p.size == 1:
                    v = float(p.item())
                else:
                    return False
            else:
                v = float(p)
        except (TypeError, ValueError):
            return False
        return math.isfinite(v) and 0.0 <= v <= 1.0

    def is_finite_non_negative(v: float) -> bool:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return False
        return math.isfinite(x) and x >= 0.0

    def clamp_probability(p: float) -> float:
        if not math.isfinite(p):
            if math.isnan(p):
                return 0.0
            return 0.0 if p < 0.0 else 1.0
        if p < 0.0:
            return 0.0
        if p > 1.0:
            return 1.0
        return p

    def clamp_probabilities(arr) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        a = np.where(np.isfinite(a), a, 0.0)
        a = np.clip(a, 0.0, None)
        s = float(a.sum())
        if s > 0.0:
            a = a / s
        return a

    def is_close(a: float, b: float, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
        return abs(float(a) - float(b)) <= max(rel_tol * max(abs(float(a)), abs(float(b))), abs_tol)

    def sanitize_for_serialization(obj):
        return obj

    def calculate_visibility(dm: float, dpsi1: float, dpsi2: float) -> float:
        return 1.0 - float(dm)

    def calculate_intensity_mismatch_factor(dm: float, dpsi1: float, dpsi2: float) -> float:
        return 1.0

    def calculate_wdm_fwm_noise(**kwargs) -> float:
        return 0.0


logger = logging.getLogger(__name__)

# === Constants / Internal Codes =============================================

STATE_VERSION = 19
_LEGACY_STATE_VERSIONS = (9, 10, 11, 12, 13, 14, 15, 16, 17, 18)

_PHYSICS_CODE_VERSION = 10

# --- Status codes returned by the event-processing kernel -------------------
STATUS_OK = 0
STATUS_RNG_EXHAUSTED = 1
STATUS_OUTPUT_BUFFER_EXHAUSTED = 2
STATUS_CARRY_OVER_BUFFER_EXHAUSTED = 3
STATUS_ITERATION_CAP_EXCEEDED = 4

# --- Diagnostic indices -----------------------------------------------------
DIAG_DEADTIME = 0          # events suppressed by dead time
DIAG_AFTERPULSE = 1        # afterpulse-originated fires
DIAG_DARKCOUNT = 2         # dark-count-originated fires
DIAG_TOSSED_CORE = 3       # dark/AP events past batch end (not carried)
DIAG_IMD_CLICKS = 4        # IMD/FWM-originated click events
DIAG_CARRY_OVER = 5        # input events carried to next batch
DIAG_DEADTIME_DROPPED = 6  # input events dropped because they fell in dead time
DIAG_MULTIPLE_FIRES_PER_SLOT = 7
DIAG_COUNT = 8

# --- ``const_params`` named indices -----------------------------------------
CP_DEAD_TIME = 0
CP_AP_PROB = 1
CP_AP_LIFETIME = 2
CP_PULSE_PERIOD = 3
CP_TIME_START = 4
CP_BATCH_DURATION = 5
CP_DR0 = 6
CP_DR1 = 7
CP_AP_GEOMETRIC = 8
CP_MAX_CASCADE = 9
CP_GATED = 10
CP_GATE_WIDTH = 11
CP_GATE_OFFSET = 12
CP_SHARED_SPAD = 13
CP_AP_RESCHEDULE_UNIFORM = 14
CP_LEN = 15

# --- Detector simulation parameters bundle ----------------------------------
# Mirrors ChannelSimParams in channel.py: a frozen bundle of detector-derived
# quantities that main_optimized.py can use without config-dict reads,
# ensuring parameter consistency and Numba-safe access.

@dataclass(frozen=True)
class DetectorSimParams:
    """Frozen bundle of detector-derived quantities for simulation use.

    This dataclass captures the detector parameters that the simulation
    driver needs at runtime for Rogers analytic SBR, CSV output, and
    any future finite-key gain-curve pre-computation.  By freezing these
    values from a validated :class:`SinglePhotonDetector` object, we
    ensure downstream code reads from the detector's frozen state —
    not from a stale config dict.

    Construction
    ------------
    Use the factory :func:`build_detector_sim_params` rather than
    constructing directly.
    """
    det_eff_d0: float
    det_eff_d1: float
    dark_rate: float
    dark_rate_d1: float
    dead_time_ns: float
    afterpulse_prob: float
    afterpulse_lifetime_ns: float
    qber_intrinsic: float
    misalignment: float
    jitter_fwhm_ns: float
    detector_type: str  # .value of DetectorType enum


def build_detector_sim_params(detector: "SinglePhotonDetector") -> DetectorSimParams:
    """Build :class:`DetectorSimParams` from a validated detector object.

    This is the single approved path for main_optimized.py to extract
    all detector-derived quantities for the simulation loop.  Every
    parameter is read from the detector's validated, frozen state —
    not from the mutable config dict.
    """
    return DetectorSimParams(
        det_eff_d0=float(detector.det_eff_d0),
        det_eff_d1=float(detector.det_eff_d1),
        dark_rate=float(detector.dark_rate),
        dark_rate_d1=float(detector.dark_rate_d1),
        dead_time_ns=float(detector.dead_time_ns),
        afterpulse_prob=float(detector.afterpulse_prob),
        afterpulse_lifetime_ns=float(detector.afterpulse_lifetime_ns),
        qber_intrinsic=float(detector.qber_intrinsic),
        misalignment=float(detector.misalignment),
        jitter_fwhm_ns=float(detector.jitter_fwhm_ns),
        detector_type=detector.detector_type.value,
    )

# --- Tuning constants (centralized per review A10/F41) ----------------------
@dataclass(frozen=True)
class DetectorConstants:
    """Centralized tuning constants for the detector model.

    All values are documented with their physical meaning and, where
    applicable, a reference.  Changing these without re-validation
    against characterization data is not recommended.
    """

    MAX_AFTERPULSE_CASCADE: int = 32

    MAX_KERNEL_ITERATIONS: int = 10_000_000

    SPAD_DARK_ACTIVATION_EV: float = 0.15

    SNSPD_REF_WAVELENGTH_NM: float = 1550.0

    REF_EXCESS_BIAS_FALLBACK_V: float = 5.0

    DEFAULT_LINEWIDTH_NM: float = 0.1

    ROGERS_SBR_MAX_CONSTANT: float = 1.433
    ROGERS_RHO_TX_MAX_CONSTANT: float = 5.92

    ROGERS_P00_MAX_ITER: int = 5000
    ROGERS_P00_TOL: float = 1e-12

    DEFAULT_RNG_BUFFER_MULT: float = 10.0
    DEFAULT_OUT_BUFFER_MULT: float = 2.5
    DEFAULT_CO_BUFFER_MULT: float = 2.0
    RETRY_RNG_GROWTH: float = 5.0
    RETRY_OUT_GROWTH: float = 4.0
    RETRY_CO_GROWTH: float = 4.0
    MAX_RETRIES: int = 3

    AP_RESCHEDULE_SLACK_NS: float = 1e-6

    LOG_SAFE_EPS: float = 1e-300

    NEVER_FIRED_SENTINEL: float = -1e18

    NEVER_FIRED_THRESHOLD: float = -1e17

    MAX_PENDING_APS_PER_DET: int = 1

    TRACK_GATE_FIRE_DIAGNOSTICS: bool = True

    DEFAULT_RECOVERY_MODEL: str = "hard_cutoff"

    LONG_RUN_WARN_NS: float = 1e14
    LONG_RUN_RAISE_NS: float = 1e15

    SNSPD_PLANCK_MAX_TERMS: int = 200
    SNSPD_PLANCK_TERM_REL_TOL: float = 1e-20

    ROGERS_P00_DIRECT_MATRIX_K_THRESHOLD: int = 10

    SNSPD_BIAS_FACTOR_CLAMP: float = 10.0
    SNSPD_BIAS_FACTOR_ALPHA: float = 0.5
    SNSPD_BIAS_FACTOR_REF_X: float = 0.9

# Physical constants needed by SNSPD black-body scaler (R7-21).
# These are defined in the except-fallback above; ensure they exist
# even when the qkd package IS available (the try-block imports
# CONST_BOLTZMANN and CONST_ELECTRON_CHARGE but not these two).
try:
    CONST_PLANCK
    CONST_SPEED_OF_LIGHT
except NameError:
    CONST_PLANCK = 6.62607015e-34       # J*s
    CONST_SPEED_OF_LIGHT = 2.99792458e8 # m/s

_CONST = DetectorConstants()

MAX_AFTERPULSE_CASCADE = _CONST.MAX_AFTERPULSE_CASCADE
MAX_KERNEL_ITERATIONS = _CONST.MAX_KERNEL_ITERATIONS
_SPAD_DARK_ACTIVATION_EV = _CONST.SPAD_DARK_ACTIVATION_EV
_SNSPD_REF_WAVELENGTH_NM = _CONST.SNSPD_REF_WAVELENGTH_NM
_REF_EXCESS_BIAS_FALLBACK_V = _CONST.REF_EXCESS_BIAS_FALLBACK_V
DEFAULT_LINEWIDTH_NM = _CONST.DEFAULT_LINEWIDTH_NM
_ROGERS_SBR_MAX_CONSTANT = _CONST.ROGERS_SBR_MAX_CONSTANT
_ROGERS_RHO_TX_MAX_CONSTANT = _CONST.ROGERS_RHO_TX_MAX_CONSTANT
_ROGERS_P00_MAX_ITER = _CONST.ROGERS_P00_MAX_ITER
_ROGERS_P00_TOL = _CONST.ROGERS_P00_TOL

_KERNEL_MAX_ITERATIONS = int(_CONST.MAX_KERNEL_ITERATIONS)
_KERNEL_MAX_CASCADE = int(_CONST.MAX_AFTERPULSE_CASCADE)
_KERNEL_AP_RESCHEDULE_SLACK_NS = float(_CONST.AP_RESCHEDULE_SLACK_NS)
_KERNEL_LOG_SAFE_EPS = float(_CONST.LOG_SAFE_EPS)
_KERNEL_MAX_PENDING_APS_PER_DET = int(_CONST.MAX_PENDING_APS_PER_DET)
_LONG_RUN_WARN_NS = float(_CONST.LONG_RUN_WARN_NS)
_LONG_RUN_RAISE_NS = float(_CONST.LONG_RUN_RAISE_NS)
_SNSPD_PLANCK_MAX_TERMS = int(_CONST.SNSPD_PLANCK_MAX_TERMS)
_SNSPD_PLANCK_TERM_REL_TOL = float(_CONST.SNSPD_PLANCK_TERM_REL_TOL)
_ROGERS_P00_DIRECT_MATRIX_K_THRESHOLD = int(
    _CONST.ROGERS_P00_DIRECT_MATRIX_K_THRESHOLD
)

_SNSPD_BIAS_FACTOR_CLAMP = float(_CONST.SNSPD_BIAS_FACTOR_CLAMP)
_SNSPD_BIAS_FACTOR_ALPHA = float(_CONST.SNSPD_BIAS_FACTOR_ALPHA)
_SNSPD_BIAS_FACTOR_REF_X = float(_CONST.SNSPD_BIAS_FACTOR_REF_X)

_EV_TO_JOULE = CONST_ELECTRON_CHARGE

_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

ROGERS_2007_POLICY = "rogers_2007"


class APRescheduleMode(Enum):
    """afterpulse reschedule-release policy.

    When an afterpulse fires during a detector's dead-time window, the
    kernel reschedules it for a later time.  The release time depends on
    this enum:

    ``END``
        Release at ``last_fire + dead_time_ns`` (the end of the
        dead-time window).  This is the legacy default and matches the
        behavior of pre-v18 code.
    ``UNIFORM``
        Release uniformly at random in ``[last_fire + dead_time_ns,
        last_fire + dead_time_ns + afterpulse_lifetime_ns]``, i.e.,
        AFTER the dead-time window.  This is the recommended mode for
        SPADs whose afterpulse release is dominated by trap lifetimes
        that are short compared to the dead time.

        Review-v12 F2: the previous implementation released in
        ``[last_fire, last_fire + dead_time_ns]`` (inside the
        dead-time window), which caused infinite rescheduling loops
        when the AP delay was small.  The corrected release is after
        the dead-time window, which is the physically correct model.

    The previous API accepted the strings ``"end"`` and ``"uniform"``
    (or any other string, silently treated as ``"end"``).  The new API
    accepts either the enum or the corresponding string; any other
    value is rejected in ``_validate_params``.
    """

    END = "end"
    UNIFORM = "uniform"

    @classmethod
    def coerce(cls, value: Any) -> "APRescheduleMode":
        """Accept the enum, the canonical string, or raise.

        Forward-compatible: legacy string values are migrated to the
        enum.  Typos (e.g. ``"unifrom"``) raise
        :class:`ParameterValidationError` instead of silently falling
        through to ``END`` behavior.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
            raise ParameterValidationError(
                f"ap_reschedule_release must be one of "
                f"{[m.value for m in cls]!r} or an APRescheduleMode "
                f"member; got {value!r}.  Typos are no longer silently "
                f"treated as 'end'."
            )
        raise ParameterValidationError(
            f"ap_reschedule_release must be a string or APRescheduleMode; "
            f"got {type(value).__name__}={value!r}."
        )


def _is_rogers_policy(value: Any) -> bool:
    """Return True iff ``value`` selects the Rogers 2007 sifting policy.

    """
    if isinstance(value, str):
        return value == ROGERS_2007_POLICY
    # Accept a future enum member named ROGERS_2007 if/when datatypes adds it.
    # Review-v7 R7-14: use EXACT string equality on both ``value`` and
    # ``name`` to avoid accidental matches.  The previous
    # ``getattr(value, "value", None) == ROGERS_2007_POLICY`` form
    # would silently match any enum member whose ``.value`` happened
    # to equal ``"rogers_2007"``, which could be a coincidence for
    # unrelated enums (e.g. a hypothetical ``SiftingMode.ROGERS_2007``
    # in a different module).  The new form requires the enum to BE
    # a DoubleClickPolicy (or duck-typed equivalent) with the exact
    # value or name.
    try:
        # ``value.value`` access raises ``AttributeError`` for
        # non-enum inputs (e.g. numbers), which we catch below.
        v = value.value
        if v == ROGERS_2007_POLICY:
            return True
        n = value.name
        if n == "ROGERS_2007":
            return True
    except AttributeError:
        return False
    return False


def _copy_rng_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-copy a numpy RNG bit-generator state dict.
    """
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, dict):
            # The ``bit_generator.state`` dict nests a ``state`` key
            # whose value is itself a dict-of-arrays in some numpy
            # versions; recurse to be safe.
            out[k] = _copy_rng_state(v)
        else:
            out[k] = v
    return out

class _RNGStateGuard:
    """Context manager that saves and restores ``rng.bit_generator.state``.

    Parameters
    ----------
    rng : numpy.random.Generator
        The RNG whose bit-generator state will be snapshotted.
    restore_on_exit : bool, optional
        Whether to restore the snapshot on ``__exit__``.  Default
        ``True`` (preserves the pre-v18 behavior).  Setting this to
        ``False`` is equivalent to the manual-snapshot pattern: the
        caller is responsible for re-applying ``guard.state`` if a
        restore is desired.
    """

    __slots__ = ("_rng", "state", "_restored", "_restore_on_exit")

    def __init__(self, rng: np.random.Generator, *, restore_on_exit: bool = True):
        # Review-v7 R7-02: emit a DeprecationWarning on every
        # construction so callers know to migrate to the manual
        # snapshot pattern.  ``stacklevel=2`` makes the warning point
        # at the caller's ``with _RNGStateGuard(...)`` line.
        warnings.warn(
            "_RNGStateGuard is deprecated (review-v7 R7-02) because its "
            "unconditional restore-on-exit semantics caused the C1/F1 "
            "bug in review-v6.  Use _copy_rng_state(rng.bit_generator.state) "
            "with manual re-application instead.  Target removal: v19.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._rng = rng
        self.state: Dict[str, Any] = {}
        self._restored = False
        self._restore_on_exit = bool(restore_on_exit)

    def __enter__(self) -> "_RNGStateGuard":
        self.state = _copy_rng_state(self._rng.bit_generator.state)
        self._restored = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Review-v7 R7-02: only restore if ``restore_on_exit`` is True.
        # The default (True) preserves the pre-v18 behavior; callers
        # who want the manual-snapshot semantics can opt out.
        if self._restore_on_exit:
            # Restore the snapshot.  Use the snapshot (not a re-copy) so
            # that any in-place mutation of the rng's internal state by
            # the body of the ``with`` is undone.
            self._rng.bit_generator.state = _copy_rng_state(self.state)
        self._restored = True
        # Do not suppress exceptions.
        return None

_SENTINEL_T = 1e18

event_np_dtype = np.dtype([
    ("time", "f8"),
    ("det", "i4"),
    ("idx", "i8"),  # int64 (review-v8 R9-10; was i4)
    ("sig", "b1"),  # 1 = signal-originated, 0 = noise (IMD/dark/AP)
])


# === Data Structures ========================================================

@dataclass
class _DetectorState:
    """Mutable runtime state of a ``SinglePhotonDetector``.

    Sentinels:
    - ``last_abs_fire_time_ns_d{0,1}``: ``-np.inf`` means "never fired".
    - ``pending_ap_time_d{0,1}``: ``< 0`` means "no afterpulse pending".
    - ``next_dc_time_d{0,1}``: ``< 0`` means "schedule on next batch
      start" or "no dark counts" (if dark rate is zero).
    - ``ap_depth_d{0,1}``: 0 means "no afterpulse pending" (matches the
      ``pending_ap_time_d{0,1} < 0`` sentinel).  Positive values record
      the cascade depth of the currently-pending afterpulse on each
      detector.  The depth is carried across batch
      boundaries so that long runs with high ``afterpulse_prob`` no
      longer silently re-count cross-batch cascades from depth 1.
    """

    # Absolute time of the last fire event (for dead-time calculations).
    last_abs_fire_time_ns_d0: float = -np.inf
    last_abs_fire_time_ns_d1: float = -np.inf

    # Pending afterpulse release times per detector.
    pending_ap_time_d0: float = -1.0
    pending_ap_time_d1: float = -1.0

    # Cascade depth of the currently-pending afterpulse per detector
    ap_depth_d0: int = 0
    ap_depth_d1: int = 0

    # Next scheduled dark count per detector.
    next_dc_time_d0: float = -1.0
    next_dc_time_d1: float = -1.0

    carry_over_events: Tuple[Tuple[float, int, int, bool], ...] = field(default_factory=tuple)

    # Total absolute time (ns) processed by the detector state.
    total_time_processed_ns: float = 0.0

    # Running counter of afterpulse generations across batches.
    afterpulse_total_count: int = 0

    last_path: Optional[str] = None


@dataclass(frozen=True)
class DetectionDiagnostics:
    """Diagnostic counters returned alongside a ``DetectionResult``.

    All ``*_flips`` and ``*_clicks`` fields refer to *input* events before
    double-click resolution unless the name explicitly says otherwise.
    """

    # Number of pulses where both D0 and D1 fired before policy resolution.
    double_clicks_total: int = 0
    # Of those, how many were resolved (kept) on D0 / D1 (RANDOM policy).
    resolved_to_d0: int = 0
    resolved_to_d1: int = 0
    rogers_resolved_to_d0: int = 0
    kept_both: int = 0
    # Dedicated double-click-discard counter (DISCARD policy).  Populated
    # directly from the double-click-resolution step.
    double_clicks_discarded: int = 0

    # Total events dropped: boundary + kernel-tossed + double-click-discard
    # + (under Rogers) rogers-discarded.  Use ``tossed_breakdown`` for an
    # itemized view.
    tossed_events: int = 0
    # Total events lost across all discard categories.  This includes
    # kernel-tossed (dark/AP past batch end), boundary losses, pre-batch
    # jitter losses, gated losses, double-click discards (DISCARD
    # policy), and Rogers-sequence discards.  Use ``tossed_breakdown``
    # for an itemized view.
    #
    # Review-v15 F-01 fix: ``tossed_events`` now includes
    # ``double_clicks_discarded`` and ``rogers_discarded_events``
    # so that ``tossed_breakdown`` decomposes correctly without
    # ``max(0, ...)`` clamping that previously masked kernel-tossed
    # dark/AP events.
    pre_batch_lost: int = 0

    tossed_boundary: int = 0

    gated_lost: int = 0

    carry_over_events: int = 0

    deadtime_dropped: int = 0

    combined_flips_with_arrivals: int = 0

    qber_flips_only: int = 0
    qber_flips: int = 0               # signal pulses whose target was QBER-flipped
    dead_time_suppressions: int = 0   # events suppressed by dead time (in-batch)
    isi_events: int = 0               # events landing outside their nominal slot
    afterpulse_events: int = 0        # afterpulse-originated fires
    dark_count_events: int = 0        # dark-count-originated fires
    imd_click_events: int = 0         # IMD/FWM-originated clicks (surviving)

    imd_absorbed_by_signal: int = 0


    imd_absorbed_by_signal_pre_jitter: int = 0
    imd_absorbed_by_signal_post_jitter: int = 0

    multiple_fires_per_slot: int = 0

    rogers_discarded_events: int = 0

    # Review-v19 F-11 fix: kernel_tossed is now populated directly from
    # DIAG_TOSSED_CORE instead of computed by subtraction in
    # tossed_breakdown.  The previous code computed:
    #   kernel_tossed = tossed_events - pre_batch_lost -
    #       tossed_boundary - gated_lost -
    #       double_clicks_discarded - rogers_discarded_events
    # which could go negative if any component was over-counted,
    # masking bugs in diagnostic accounting.  The new field is
    # populated directly from the kernel's DIAG_TOSSED_CORE diagnostic.
    kernel_tossed: int = 0

    in_gate_fires: int = 0
    out_of_gate_fires: int = 0

    # Buffer / RNG health flags.
    rng_exhausted: bool = False
    buffer_resized: bool = False
    num_retries: int = 0

    @property
    def misalignment_flips(self) -> int:
        """Deprecated alias for :attr:`combined_flips_with_arrivals`.

        .. deprecated:: v18 (review-v7 R7-19)
            The field actually counted combined (misalignment XOR QBER)
            flips where arrivals > 0, not pure misalignment flips.  Use
            :attr:`combined_flips_with_arrivals` for the same value
            under a clear name.  This property emits a
            :class:`DeprecationWarning` on every access and will be
            removed in v19.
        """
        warnings.warn(
            "DetectionDiagnostics.misalignment_flips is deprecated "
            "(review-v7 R7-19).  Use combined_flips_with_arrivals for "
            "the same value under a clear name.  Target removal: v19.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.combined_flips_with_arrivals

    def to_tally_counts(self) -> TallyCounts:
        """Map to the project-wide ``TallyCounts`` summary.
        """
        return TallyCounts(double_clicks_discarded=int(self.double_clicks_discarded))

    METADATA_KEY_PREFIX = "diag_"

    @classmethod
    def metadata_keys(cls, *, prefix: str = METADATA_KEY_PREFIX) -> List[str]:
        """Canonical ordered list of flat metadata keys produced by
        :meth:`to_metadata_dict`.

        """
        return [prefix + f.name for f in dataclass_fields(cls)]

    def to_metadata_dict(self, *, prefix: str = METADATA_KEY_PREFIX) -> Dict[str, Any]:
        """Flatten into a ``{<prefix><field>: value}`` dict for downstream
        consumers (CSV rows, parquet schemas, ...).

        """
        return {
            prefix + f.name: getattr(self, f.name)
            for f in dataclass_fields(self)
        }

    @property
    def tossed_breakdown(self) -> Dict[str, int]:
        """Itemize ``tossed_events`` by category.

        Returns a dict with keys: ``kernel_tossed`` (dark/AP events
        past batch end, counted in ``DIAG_TOSSED_CORE``),
        ``pre_batch_lost`` (jitter pushed event before batch),
        ``tossed_boundary`` (events past the batch end after the
        binning-stage ``valid_mask`` filter), ``gated_lost`` (events
        whose jittered time fell outside the gate window for their
        pulse slot), ``double_clicks_discarded`` (DISCARD policy),
        ``rogers_discarded_events`` (Rogers policy),
        ``deadtime_dropped`` (input events in dead time past batch
        end; informational-only -- NOT in ``tossed_events``, but
        reported here for transparency).

        ``carry_over_events`` is reported separately (it is NOT in
        ``tossed_events`` -- those events are preserved for the next
        batch, not lost).

        Review-v15 F-01 fix: ``tossed_events`` now includes
        ``double_clicks_discarded`` and ``rogers_discarded_events``,
        so the decomposition is exact (no ``max(0, ...)`` clamp).
        The previous code subtracted these from ``tossed_events``
        even though they were never added, causing ``kernel_tossed``
        to be clamped to zero and masking dark/AP events past batch
        end.

        Review-v19 F-11 fix: ``kernel_tossed`` is now populated
        directly from ``DIAG_TOSSED_CORE`` in the kernel diagnostic
        (stored in the ``kernel_tossed`` field of this class).  The
        previous code computed ``kernel_tossed`` by subtraction from
        ``tossed_events``, which could go negative if any component
        was over-counted, masking bugs.  The direct approach is
        more robust.
        """
        # Review-v19 F-11: use direct field instead of subtraction.
        kernel_tossed = self.kernel_tossed
        return {
            "kernel_tossed": int(kernel_tossed),
            "pre_batch_lost": int(self.pre_batch_lost),
            "tossed_boundary": int(self.tossed_boundary),
            "gated_lost": int(self.gated_lost),
            "deadtime_dropped": int(self.deadtime_dropped),
            "carry_over_events": int(self.carry_over_events),
            "double_clicks_discarded": int(self.double_clicks_discarded),
            "rogers_discarded_events": int(self.rogers_discarded_events),
        }

    @property
    def total_discarded(self) -> int:
        """Sum of all discard categories.

        Includes ``tossed_events`` plus ``deadtime_dropped`` (the
        latter is informational-only and not in ``tossed_events`` by
        default).  Use this for the "everything that was lost" count.
        """
        return int(self.tossed_events + self.deadtime_dropped)


class DetectionResult(NamedTuple):
    """Output of :meth:`SinglePhotonDetector.simulate_detection`.

    For threshold detectors (``SPD``/``SNSPD``), ``click0``/``click1`` are
    ``np.bool_`` arrays of length ``num_pulses`` indicating whether the
    detector fired in that pulse slot.  For ``PNRD`` detectors they are
    integer count arrays (``np.int64``).

    ``state_snapshot`` is a JSON string produced by
    :meth:`SinglePhotonDetector.get_state`; ``None`` only when the caller
    explicitly disables state serialization via ``serialize_state=False``
    (review F25).
    """

    click0: NDArray[Any]
    click1: NDArray[Any]
    diagnostics: Optional[DetectionDiagnostics] = None
    state_snapshot: Optional[str] = None


class PhotonResolvedResult(NamedTuple):
    """Photon-number-resolved simulation output (review-v4 F-42).

    Returned by :meth:`SinglePhotonDetector.simulate_detection` when
    ``return_photon_resolved=True``.  In addition to the standard
    ``click0``/``click1`` arrays, exposes per-pulse photon arrivals and
    target-detector masks so callers can compute decoy-state yields
    ``Y_n = P(click | n photons)`` directly.
    The ``arrivals`` array is the photon count after channel thinning
    (Binomial with effective transmittance) but BEFORE detector
    efficiency.  ``target_d0`` is the per-pulse target-detector mask
    AFTER misalignment+QBER flips.  ``ideal_target_d0`` is the
    corresponding mask BEFORE flips.
    """

    click0: NDArray[Any]
    click1: NDArray[Any]
    arrivals: NDArray[Any]
    target_d0: NDArray[Any]
    ideal_target_d0: NDArray[Any]
    diagnostics: Optional[DetectionDiagnostics] = None
    state_snapshot: Optional[str] = None


class AfterpulseModel(Enum):
    """Delay distribution model for afterpulse release times.

    ``EXPONENTIAL``
        Continuous exponential delay with mean ``afterpulse_lifetime_ns``.
    ``GEOMETRIC``
        Discrete delay on the pulse-period grid.  The geometric parameter
        ``p`` is derived from ``afterpulse_lifetime_ns`` and the
        per-batch ``pulse_period_ns`` via ``p = 1 - exp(-T/tau)`` so that
        ``afterpulse_lifetime_ns`` remains the characteristic time in
        both modes.
    """

    GEOMETRIC = "geometric"
    EXPONENTIAL = "exponential"


class DeadTimeModel(Enum):
    """Dead-time model selector.

    Per Rogers et al. (2007) §3 / §6, **individual** SPADs are
    **non-paralyzable**: a photon arriving during the dead time has no
    effect on the detector and does NOT extend the dead-time window.
    Paralyzability in the Rogers sense is an *emergent* property of the
    **basis-level pair of detectors** together with the
    sequence-collapsing sifting rule (see :class:`DoubleClickPolicy`'s
    ``ROGERS_2007`` value).

    ``PARALYZABLE`` is retained for backward compatibility.  In
    ``strict_mode`` (the default), selecting it raises
    :class:`ParameterValidationError`.  In non-strict mode it is treated
    identically to ``NON_PARALYZABLE`` with a warning.  To reproduce the
    paralyzable *basis-level* behavior of Rogers et al., use
    ``NON_PARALYZABLE`` here and select ``ROGERS_2007_POLICY``.
    """

    NON_PARALYZABLE = "non_paralyzable"
    PARALYZABLE = "paralyzable"


class DetectorTopology(Enum):
    """Detector-pair topology selector.

    ``INDEPENDENT_SPADS``
        Two independent SPADs, each with its own dead-time clock.
        This is the standard BB84 setup with a polarizing beam splitter
        feeding two SPADs.  The dead-time check is per-detector
        (``last_fire_d0`` vs ``last_fire_d1``).
    ``SHARED_SPAD``
        A single SPAD with optical routing (e.g. a 2-to-1 fiber switch
        or a polarization-insensitive SPAD with subsequent polarizing
        optics).  The dead-time clock is SHARED: any fire on either
        channel blocks BOTH channels until ``dead_time_ns`` has elapsed.
        This is the physically correct model for receivers that use a
        single SPAD with after-the-fact polarization analysis.
    """

    INDEPENDENT_SPADS = "independent_spads"
    SHARED_SPAD = "shared_spad"


class CarryOverEvent(NamedTuple):
    """Structured carry-over event (review-v5 F-49).

    Replaces the previous bare ``(time_abs_ns, det_id, src_pulse_idx,
    is_signal_origin)`` tuple.  Field names make the carry-over event
    self-documenting and protect against accidental re-ordering when
    the tuple shape changes.
    """

    time_abs_ns: float
    det_id: int
    src_pulse_idx: int
    is_signal_origin: bool


# === Decoders ===============================================================

@dataclass(frozen=True, slots=True)
class EntanglingDecoder:
    """Entangling-decoder helper with visibility-aware dephasing.

    The dephasing map applied here is the **standard** CPTP dephasing
    channel on the joint Hilbert space: off-diagonal entries of
    ``rho_joint`` are scaled by ``visibility`` (the interference
    visibility), while the diagonal (populations) is preserved.
    Positivity and unit-trace are checked after the map.
    """

    redundancy_M: int
    architecture: DecoderArchitecture = DecoderArchitecture.ENTANGLING

    physical_gate_efficiency: float = 1.0
    interference_visibility: float = 1.0
    gate_success_prob: float = 0.5

    # When True, raw probability vectors whose sum deviates from 1 by more
    # than ``raw_sum_tol`` raise instead of being silently renormalized
    # by ``clamp_probabilities`` (review F46/N8).
    strict_normalization: bool = True
    raw_sum_tol: float = 1e-9

    def __post_init__(self):
        if self.architecture != DecoderArchitecture.ENTANGLING:
            raise ParameterValidationError(
                "EntanglingDecoder must use DecoderArchitecture.ENTANGLING"
            )
        if self.redundancy_M not in (2, 3):
            raise ParameterValidationError(
                "EntanglingDecoder currently supports redundancy_M in {2, 3}."
            )
        for name in ("physical_gate_efficiency", "interference_visibility", "gate_success_prob"):
            value = getattr(self, name)
            if not is_valid_probability(value):
                raise ParameterValidationError(
                    f"Parameter '{name}' must be a valid probability in [0, 1]."
                )
        if self.raw_sum_tol <= 0.0 or not math.isfinite(self.raw_sum_tol):
            raise ParameterValidationError(
                "Parameter 'raw_sum_tol' must be finite and strictly positive."
            )

    def decode_probabilities(
        self, rho_list: List[np.ndarray]
    ) -> Tuple[float, ...]:
        """Return ``(p_outcome..., p_failure)`` probabilities.

        The last element of the returned tuple is the *failure*
        probability, i.e. ``1 - efficiency * sum(decoded_probs)``.  This
        makes the failure outcome explicit instead of implicitly encoded
        as a sub-unity sum.
        """
        if len(rho_list) == 0:
            # Redundancy-2 path: 2 outcomes + failure = 3-tuple.
            # Redundancy-3 path: 4 outcomes + failure = 5-tuple.
            n_outcomes = 2 if self.redundancy_M == 2 else 4
            return tuple([0.0] * n_outcomes) + (1.0,)

        rho_joint = self._validate_and_build_rho(rho_list)

        if not is_close(self.interference_visibility, 1.0):
            rho_joint = self._apply_dephasing(rho_joint, self.interference_visibility)

        # Renormalize against numerical drift after dephasing.
        tr = np.trace(rho_joint).real
        if tr > EPS:
            rho_joint = rho_joint / tr

        p_gate = self.gate_success_prob * self.physical_gate_efficiency

        if self.redundancy_M == 2:
            outcomes = self._decode_bell_states(rho_joint)
        else:
            outcomes = self._decode_3qubit_2bit(rho_joint)

        outcomes_arr = np.asarray(outcomes, dtype=float)
        outcomes_tuple = self._clamp_and_check(outcomes_arr)

        scaled = np.asarray(outcomes_tuple, dtype=float) * p_gate
        scaled = np.clip(scaled, 0.0, 1.0)
        p_success = float(np.sum(scaled))
        p_failure = max(0.0, 1.0 - p_success)
        return tuple(scaled.tolist()) + (p_failure,)

    # --- Internal helpers ---

    @staticmethod
    def _validate_and_build_rho(rho_list: List[np.ndarray]) -> np.ndarray:
        """Validate input density matrices and build the joint rho.
        """
        if not isinstance(rho_list, (list, tuple)):
            raise ParameterValidationError("rho_list must be a list/tuple of arrays.")
        rho_joint = None
        hermitian_tol = 1e-9
        psd_tol = -1e-9  # allow tiny negative eigenvalues from rounding
        for i, r in enumerate(rho_list):
            arr = np.asarray(r, dtype=complex)
            if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
                raise ParameterValidationError(
                    f"rho_list[{i}] must be a square 2D matrix; got shape {arr.shape}."
                )
            if arr.shape[0] == 0:
                raise ParameterValidationError(f"rho_list[{i}] is empty.")
            if not np.allclose(arr, arr.conj().T, atol=hermitian_tol):
                max_diff = float(np.max(np.abs(arr - arr.conj().T)))
                raise ParameterValidationError(
                    f"rho_list[{i}] is not Hermitian (max |rho - rho^dagger| "
                    f"= {max_diff:.4g} > {hermitian_tol:.4g}).  Density "
                    f"matrices must be Hermitian; check the input "
                    f"construction (F-29)."
                )
            try:
                eigvals = np.linalg.eigvalsh(0.5 * (arr + arr.conj().T))
            except np.linalg.LinAlgError as e:
                raise ParameterValidationError(
                    f"rho_list[{i}] eigenvalue computation failed: {e} (F-29)."
                ) from e
            min_eig = float(np.min(eigvals))
            if min_eig < psd_tol:
                raise ParameterValidationError(
                    f"rho_list[{i}] is not positive semidefinite "
                    f"(min eigenvalue = {min_eig:.4g} < {psd_tol:.4g}).  "
                    f"Density matrices must be PSD; check the input "
                    f"construction (F-29)."
                )
            rho_joint = arr if rho_joint is None else np.kron(rho_joint, arr)
        return rho_joint

    @staticmethod
    def _apply_dephasing(rho: np.ndarray, visibility: float) -> np.ndarray:
        """Standard CPTP dephasing channel: scale coherences by ``visibility``."""
        n = rho.shape[0]
        mask = np.eye(n, dtype=complex)
        off_diag = 1.0 - mask
        out = rho * mask + rho * off_diag * float(visibility)
        # Symmetrize Hermitian part against numerical noise.
        out = 0.5 * (out + out.conj().T)
        return out

    def _clamp_and_check(self, raw: np.ndarray) -> Tuple[float, ...]:
        """Clamp probabilities to [0,1] and renormalize.

        Review F46/N8: when ``strict_normalization`` is True, raise if
        the raw sum deviates from 1 by more than ``raw_sum_tol``.
        """
        # Review-v7 R7-17: per-element strict check BEFORE clipping.
        if self.strict_normalization:
            arr = np.asarray(raw, dtype=float)
            max_val = float(np.max(arr)) if arr.size > 0 else 0.0
            min_val = float(np.min(arr)) if arr.size > 0 else 0.0
            if max_val > 1.0 + self.raw_sum_tol:
                raise ParameterValidationError(
                    f"Raw outcome probability {max_val!r} exceeds 1.0 by "
                    f"more than raw_sum_tol={self.raw_sum_tol} "
                    f"(review-v7 R7-17).  This indicates the input "
                    f"density matrix has trace > 1 or the Bell-state "
                    f"projector computed an out-of-range probability.  "
                    f"Check the input density matrix normalization."
                )
            if min_val < -self.raw_sum_tol:
                raise ParameterValidationError(
                    f"Raw outcome probability {min_val!r} is below 0 by "
                    f"more than raw_sum_tol={self.raw_sum_tol} "
                    f"(review-v7 R7-17).  This indicates the input "
                    f"density matrix is not positive semidefinite or "
                    f"the projector computation introduced significant "
                    f"numerical error.  Check the input density matrix."
                )
        raw_sum = float(np.sum(raw))
        if self.strict_normalization and abs(raw_sum - 1.0) > self.raw_sum_tol:
            raise ParameterValidationError(
                f"Raw outcome probabilities sum to {raw_sum!r}, expected 1.0 "
                f"(tol={self.raw_sum_tol}). This indicates a bug in the "
                f"Bell-state projector or input density matrix."
            )
        return tuple(clamp_probabilities(raw).tolist())

    @staticmethod
    def _decode_bell_states(rho_joint: np.ndarray) -> Tuple[float, float]:
        psi_minus = np.zeros((4, 1), dtype=complex)
        psi_minus[1, 0] = 1.0 / math.sqrt(2.0)
        psi_minus[2, 0] = -1.0 / math.sqrt(2.0)
        p_minus = psi_minus @ psi_minus.conj().T

        psi_plus = np.zeros((4, 1), dtype=complex)
        psi_plus[1, 0] = 1.0 / math.sqrt(2.0)
        psi_plus[2, 0] = 1.0 / math.sqrt(2.0)
        p_plus = psi_plus @ psi_plus.conj().T

        raw = np.array(
            [
                np.real(np.trace(p_minus @ rho_joint)),
                np.real(np.trace(p_plus @ rho_joint)),
            ],
            dtype=float,
        )
        return tuple(raw.tolist())

    @staticmethod
    def _decode_3qubit_2bit(rho_joint: np.ndarray) -> Tuple[float, float, float, float]:
        # 3-qubit bit-flip code: the four code-word states are
        # |000>, |011>, |101>, |110> (indices 0, 3, 5, 6 in the
        # computational basis).  Each encodes a 2-bit syndrome.
        # Reference: Nielsen & Chuang, Quantum Computation and Quantum
        # Information, Chapter 10 (quantum error correction).
        p_00 = np.zeros((8, 8)); p_00[0, 0] = 1.0   # |000><000|
        p_01 = np.zeros((8, 8)); p_01[3, 3] = 1.0   # |011><011|
        p_10 = np.zeros((8, 8)); p_10[5, 5] = 1.0   # |101><101|
        p_11 = np.zeros((8, 8)); p_11[6, 6] = 1.0   # |110><110|

        raw = np.array(
            [
                np.real(np.trace(p_00 @ rho_joint)),
                np.real(np.trace(p_01 @ rho_joint)),
                np.real(np.trace(p_10 @ rho_joint)),
                np.real(np.trace(p_11 @ rho_joint)),
            ],
            dtype=float,
        )
        return tuple(raw.tolist())


# === Detector ===============================================================

@dataclass(slots=True)
class SinglePhotonDetector:
    """Stateful single-photon detector model.

    The detector supports three operating modes:

    - ``DetectorType.SPD``  -- SPAD threshold detector, event-driven.
    - ``DetectorType.SNSPD`` -- SNSPD threshold detector, event-driven
      with SNSPD-specific dark-rate scaling (Planck-spectrum black-body
      model plus bias-current scaling).
    - ``DetectorType.PNRD`` -- photon-number-resolving path, per-pulse
      binomial thinning, no continuous-time state.

    All units are SI unless stated otherwise.  Dark-rate is in Hz;
    ``pulse_period_ns`` is supplied per call to
    :meth:`simulate_detection`.
    """

    det_eff_d0: float
    det_eff_d1: float

    dark_rate: float = field(metadata={"noise": True})      # Hz
    qber_intrinsic: float = field(metadata={"noise": True})
    misalignment: float = field(metadata={"noise": True})
    double_click_policy: DoubleClickPolicy
    detector_type: DetectorType = DetectorType.SPD

    dead_time_ns: float = field(default=0.0, metadata={"noise": True})
    jitter_fwhm_ns: float = field(default=0.0, metadata={"noise": True})

    jitter_fwhm_ns_d0: Optional[float] = field(default=None, metadata={"noise": True})
    jitter_fwhm_ns_d1: Optional[float] = field(default=None, metadata={"noise": True})

    dead_time_model: DeadTimeModel = DeadTimeModel.NON_PARALYZABLE
    afterpulse_model: AfterpulseModel = AfterpulseModel.EXPONENTIAL

    # Afterpulse parameters.
    afterpulse_prob: float = field(default=0.0, metadata={"noise": True})
    afterpulse_lifetime_ns: float = 100.0

    # Hardware / environmental parameters.
    temperature_k: float = 293.0
    bias_voltage: float = 50.0
    breakdown_voltage: float = 45.0

    ref_temperature_k: float = 293.0
    ref_bias_voltage: float = 50.0

    bias_current: Optional[float] = None
    switching_current: Optional[float] = None

    gated: bool = False
    gate_width_ns: float = 1.0

    gate_offset_ns: float = 0.0

    dark_rate_d1: Optional[float] = field(default=None, metadata={"noise": True})

    det_eff_d0_z: Optional[float] = None
    det_eff_d0_x: Optional[float] = None
    det_eff_d1_z: Optional[float] = None
    det_eff_d1_x: Optional[float] = None

    detector_topology: DetectorTopology = DetectorTopology.INDEPENDENT_SPADS

    ap_reschedule_release: APRescheduleMode = APRescheduleMode.END

    temperature_k_d0: Optional[float] = None
    temperature_k_d1: Optional[float] = None
    bias_voltage_d0: Optional[float] = None
    bias_voltage_d1: Optional[float] = None
    # Review-v17 H2: per-detector breakdown voltage overrides.  When
    # None, fall back to the shared ``breakdown_voltage``.  Required
    # for modeling asymmetric SPADs with manufacturing variation or
    # temperature-gradient-induced breakdown differences.
    breakdown_voltage_d0: Optional[float] = None
    breakdown_voltage_d1: Optional[float] = None

    # Review-v19 F-01 fix: temperature-dependent breakdown voltage
    # coefficient for InGaAs SPADs.  Breakdown voltage shifts at
    # approximately 50 mV/K for typical InGaAs SPADs.  When
    # ``ref_temperature_k != temperature_k``, the operating-point
    # breakdown voltage is adjusted: ``bdv_op = breakdown_voltage +
    # breakdown_temp_coeff_mv_per_k * 1e-3 * (temperature_k -
    # ref_temperature_k)``.  The previous code used the same
    # ``breakdown_voltage`` for both operating and reference excess-bias,
    # silently producing wrong values when temperatures differ.  Set to
    # 0.0 to disable the correction (equivalent to the old behavior).
    # Default is 50.0 mV/K (typical InGaAs).
    breakdown_temp_coeff_mv_per_k: float = 50.0


    allow_qualitative_snspd_scaler: bool = False

    recovery_time_ns: float = 0.0

    max_count_rate_hz: Optional[float] = None

    max_pending_aps_per_det: int = _CONST.MAX_PENDING_APS_PER_DET

    strict_mode: bool = True

    # Internal state (excluded from config serialization).
    _internal_state: _DetectorState = field(init=False, repr=False, compare=False)
    # Review-v18 F-03: transient per-call cache for _calculate_dynamic_dark_rates.
    # Not serialized, not compared.  Initialized to None; invalidated at the
    # start of each simulate_detection call.
    _cached_dynamic_dark_rates: Optional[Tuple[float, float]] = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self):
        if not isinstance(self.ap_reschedule_release, APRescheduleMode):
            coerced = APRescheduleMode.coerce(self.ap_reschedule_release)
            object.__setattr__(self, "ap_reschedule_release", coerced)
        if self.max_pending_aps_per_det != _CONST.MAX_PENDING_APS_PER_DET:
            raise ParameterValidationError(
                f"max_pending_aps_per_det={self.max_pending_aps_per_det} "
                f"is not supported: the kernel currently models only "
                f"{_CONST.MAX_PENDING_APS_PER_DET} pending AP per "
                f"detector (F-07).  The multi-pending-AP queue is a "
                f"future-work item; see the module docstring."
            )
        self._validate_params()
        self._internal_state = self._create_initial_state()

    # ---- Construction / serialization ----

    @classmethod
    def from_config(
        cls,
        config: DetectionConfig,
        qber_intrinsic: float = 0.0,
        misalignment: float = 0.0,
        double_click_policy: DoubleClickPolicy = DoubleClickPolicy.RANDOM,
        **kwargs,
    ) -> "SinglePhotonDetector":
        """Build a detector from a :class:`DetectionConfig` plus extra
        physical/runtime parameters supplied via ``kwargs``.

        Mapping:
        - ``config.efficiency`` -> both detector efficiencies,
        - ``config.dark_count_rate`` -> base dark rate (Hz),
        - ``config.detector_type`` -> detector type.
        """
        # Validate kwargs against dataclass fields (review F40).
        allowed = {f.name for f in dataclass_fields(cls) if f.init}
        unknown = sorted(set(kwargs.keys()) - allowed)
        if unknown:
            raise ConfigurationError(
                f"Unknown detector parameter(s) passed via kwargs: {unknown}. "
                f"Allowed: {sorted(allowed)}."
            )

        return cls(
            det_eff_d0=config.efficiency,
            det_eff_d1=config.efficiency,
            dark_rate=config.dark_count_rate,
            detector_type=config.detector_type,
            qber_intrinsic=qber_intrinsic,
            misalignment=misalignment,
            double_click_policy=double_click_policy,
            **kwargs,
        )

    @classmethod
    def from_config_dict(cls, config_dict: Dict[str, Any], **extra_kwargs) -> "SinglePhotonDetector":
        """Strict construction from a plain dict.

        Unknown keys are rejected.  Required keys mirror the dataclass
        fields without defaults.  Enum-valued fields are coerced from
        their string/enum value.
        """
        if not isinstance(config_dict, dict):
            raise ConfigurationError("Detector config must be a dictionary.")

        required = {
            "det_eff_d0", "det_eff_d1", "dark_rate",
            "qber_intrinsic", "misalignment", "double_click_policy",
        }
        missing = sorted(required - set(config_dict.keys()))
        if missing:
            raise ConfigurationError(
                f"Missing required detector config keys: {missing}"
            )

        allowed = {f.name for f in dataclass_fields(cls) if f.init}
        unknown = sorted(set(config_dict.keys()) - allowed)
        if unknown:
            raise ConfigurationError(
                f"Unknown detector config keys: {unknown}"
            )

        normalized = dict(config_dict)
        # Merge extra kwargs (e.g. flip_prob_override, allow_xor_flip_formula)
        # AFTER the unknown-key check so they bypass the strict filter but
        # still flow into cls(**normalized) → __init__'s **kwargs handler.
        # These are constructor-only kwargs that aren't dataclass fields.
        normalized.update(extra_kwargs)

        enum_fields = {
            "double_click_policy": DoubleClickPolicy,
            "detector_type": DetectorType,
            "dead_time_model": DeadTimeModel,
            "afterpulse_model": AfterpulseModel,
        }
        for key, enum_cls in enum_fields.items():
            if key not in normalized:
                continue
            value = normalized[key]
            if isinstance(value, enum_cls):
                continue
            # The ROGERS_2007 sentinel is a plain string and is accepted
            # as-is for ``double_click_policy``.
            if key == "double_click_policy" and _is_rogers_policy(value):
                continue
            try:
                normalized[key] = enum_cls(value)
            except Exception as e:
                raise ConfigurationError(
                    f"Invalid value for enum field '{key}': {value!r} "
                    f"(original error: {e})"
                ) from e

        try:
            return cls(**normalized)
        except ParameterValidationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to build detector from config: {e}") from e

    def to_config_dict(self, *, include_experimental: bool = True) -> Dict[str, Any]:
        """Serialize configuration to a plain dict.

        Parameters
        ----------
        include_experimental:
            When ``False``, runtime/experimental fields
            (``strict_mode``, ``temperature_k``, ``bias_voltage``,
            ``breakdown_voltage``, ``ref_temperature_k``,
            ``ref_bias_voltage``, ``bias_current``,
            ``switching_current``, ``gated``, ``gate_width_ns``) are
            omitted.  This is the recommended mode for stable
            cross-version serialization of the "physics" config; the
            experimental fields are still available when explicitly
            requested.
        """
        experimental = {
            "strict_mode",
            "temperature_k",
            "bias_voltage",
            "breakdown_voltage",
            "ref_temperature_k",
            "ref_bias_voltage",
            "bias_current",
            "switching_current",
            "gated",
            "gate_width_ns",
            "gate_offset_ns",
        }
        config: Dict[str, Any] = {}
        for f in dataclass_fields(self):
            if f.name == "_internal_state":
                continue
            if not include_experimental and f.name in experimental:
                continue
            value = getattr(self, f.name)
            config[f.name] = value.value if isinstance(value, Enum) else value
        return sanitize_for_serialization(config)

    def reset_state(self):
        """Reset runtime state to initial conditions."""
        self._internal_state = self._create_initial_state()

    def get_state(self) -> str:
        """Serialize runtime state to a JSON string.

        Review C2/F2 fix: ``last_fire_d0`` / ``last_fire_d1`` are
        serialized as a JSON-safe sentinel (``NEVER_FIRED_SENTINEL``,
        a finite ``-1e18``) when their internal value is ``-np.inf``
        (the "never fired" sentinel used by :class:`_DetectorState`).
        Previously the raw ``-np.inf`` was passed to ``json.dumps``,
        which raises ``ValueError: Out of range float values are not
        JSON compliant`` under the default ``allow_nan=False`` setting.
        This made state serialization crash for any detector that had
        never fired on at least one channel -- an extremely common
        situation (initial state, low-click channel, ...).
        """
        state = self._internal_state

        def _safe_fire_time(v: float) -> float:
            """Map ``-np.inf`` to a finite JSON-safe sentinel (review C2).

            Review-v6 F39 fix: raise ``ConfigurationError`` on ``+inf``.
            ``+inf`` should never occur for fire times (fire times are
            absolute physical timestamps bounded by the batch end).  The
            previous code silently mapped ``+inf`` to ``+1e18`` (a finite
            positive sentinel) on serialization, then ``_restore_fire_time``
            returned ``1e18`` (NOT ``+inf``) on deserialization -- a
            LOSSY round-trip that masked upstream bugs.  The fix is to
            raise eagerly so the upstream bug is surfaced.
            """
            fv = float(v)
            if math.isfinite(fv):
                return fv
            # ``+inf`` is a bug -- raise instead of silently mapping.
            if fv > 0.0:
                raise ConfigurationError(
                    f"last_fire time = +inf is not a valid fire time "
                    f"(fire times must be finite or -inf for 'never "
                    f"fired').  This indicates a bug in the kernel or "
                    f"upstream state manipulation (review-v6 F39)."
                )
            # ``NaN`` is likewise never a valid fire time -- it signals an
            # upstream arithmetic bug (e.g. 0/0 or inf-inf in the kernel).
            # Raise eagerly, mirroring the ``+inf`` branch, instead of
            # silently mapping it to the "never fired" sentinel (which
            # would be a lossy round-trip that masks the bug).
            if math.isnan(fv):
                raise ConfigurationError(
                    f"last_fire time = NaN is not a valid fire time "
                    f"(fire times must be finite or -inf for 'never "
                    f"fired').  This indicates a bug in the kernel or "
                    f"upstream state manipulation."
                )
            # Only ``-inf`` ("never fired") remains -> JSON-safe sentinel.
            return _CONST.NEVER_FIRED_SENTINEL  # -1e18

        payload = {
            "version": STATE_VERSION,
            "config_hash": self._config_hash(),
            "code_version": _PHYSICS_CODE_VERSION,
            "last_fire_d0": _safe_fire_time(state.last_abs_fire_time_ns_d0),
            "last_fire_d1": _safe_fire_time(state.last_abs_fire_time_ns_d1),
            "pending_ap_d0": state.pending_ap_time_d0,
            "pending_ap_d1": state.pending_ap_time_d1,
            "ap_depth_d0": int(state.ap_depth_d0),
            "ap_depth_d1": int(state.ap_depth_d1),
            "next_dc_time_d0": state.next_dc_time_d0,
            "next_dc_time_d1": state.next_dc_time_d1,
            "carry_over": state.carry_over_events,
            "total_time_processed_ns": state.total_time_processed_ns,
            "afterpulse_total_count": state.afterpulse_total_count,
            "last_path": state.last_path,
            "rng_buffer_mult_default": _CONST.DEFAULT_RNG_BUFFER_MULT,
        }
        return json.dumps(payload, sort_keys=True, allow_nan=False)

    def set_state(self, state_json: str):
        """Restore runtime state from a JSON string.

        Forward-migrates from legacy state versions 9, 10, 11, 12, 13, 14.
        Raises :class:`ConfigurationError` on version mismatch, config-hash
        mismatch, or physics-code-version mismatch (review F61).
        """
        try:
            d = json.loads(state_json)
            version = d.get("version")
            if version not in _LEGACY_STATE_VERSIONS + (STATE_VERSION,):
                raise ConfigurationError(
                    f"State version incompatible: got {version!r}, "
                    f"supported legacy versions are {_LEGACY_STATE_VERSIONS}, "
                    f"current is {STATE_VERSION}.  States produced by older "
                    f"code versions (1-8) cannot be loaded; please "
                    f"reset_state() and re-run.  States produced by newer "
                    f"code versions are not backward-compatible."
                )

            saved_hash = d.get("config_hash")
            current_hash = self._config_hash()
            # F-L1: config_hash is now a full 64-char SHA-256 digest.
            # Accept legacy 16-char hashes by comparing only the prefix
            # when the saved hash is shorter than the current one.
            if saved_hash is not None and saved_hash != current_hash:
                if len(saved_hash) == 16 and current_hash.startswith(saved_hash):
                    # Legacy short hash matches the prefix of the full
                    # hash; treat as compatible (same config, just a
                    # different hash length convention).
                    pass
                else:
                    raise ConfigurationError(
                        "State/config hash mismatch: the detector configuration "
                        "differs from the one that produced this state."
                    )

            saved_code_version = d.get("code_version", 0)
            # Legacy state versions 9-13 were produced by physics-code
            # versions 1-4.  Refuse to load across the v4->v5 boundary
            # (gate_offset_ns added, AP-reschedule semantics changed)
            # because the kernel would otherwise misinterpret the
            # const_params layout and the rescheduled-AP state.
            if saved_code_version != _PHYSICS_CODE_VERSION:
                raise ConfigurationError(
                    f"Physics code version mismatch: state was produced by "
                    f"code version {saved_code_version}, current is "
                    f"{_PHYSICS_CODE_VERSION}.  The kernel semantics or "
                    f"const_params layout has changed; cannot safely "
                    f"restore state.  Please reset_state() and re-run."
                )

            # Legacy versions 9/10 had a single ``next_dc_time`` field.
            # Review F62: set BOTH d0 and d1 to the legacy value as a
            # best-effort migration (previously only d0 was set).
            next_dc_d0 = d.get("next_dc_time_d0", -1.0)
            next_dc_d1 = d.get("next_dc_time_d1", -1.0)
            if "next_dc_time" in d:
                legacy_t = float(d["next_dc_time"])
                if next_dc_d0 < 0:
                    next_dc_d0 = legacy_t
                if next_dc_d1 < 0:
                    next_dc_d1 = legacy_t

            raw_co = d.get("carry_over", [])

            carry_over = tuple(
                CarryOverEvent(
                    time_abs_ns=float(t),
                    det_id=int(det),
                    src_pulse_idx=int(idx),
                    is_signal_origin=bool(sig),
                )
                for t, det, idx, sig in raw_co
            )

            for i, co in enumerate(carry_over):
                if not math.isfinite(co.time_abs_ns) or co.time_abs_ns < 0.0:
                    raise ConfigurationError(
                        f"carry_over[{i}].time_abs_ns = {co.time_abs_ns!r} "
                        f"is invalid: must be finite and non-negative (F-31)."
                    )
                if co.det_id not in (0, 1):
                    raise ConfigurationError(
                        f"carry_over[{i}].det_id = {co.det_id!r} is "
                        f"invalid: must be 0 or 1 (F-31)."
                    )
                if co.src_pulse_idx < 0:
                    raise ConfigurationError(
                        f"carry_over[{i}].src_pulse_idx = "
                        f"{co.src_pulse_idx!r} is invalid: must be "
                        f"non-negative (F-31)."
                    )

            # Review C2/F2: restore ``-np.inf`` from the JSON-safe sentinel.
            def _restore_fire_time(v) -> float:
                """Map the JSON sentinel back to ``-np.inf`` (review C2)."""
                # ``json.loads`` may yield ``-Infinity`` if the producing
                # side used ``allow_nan=True`` (legacy code path); accept
                # both representations.
                if v is None:
                    return -np.inf
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return -np.inf
                if math.isfinite(fv) and fv <= _CONST.NEVER_FIRED_THRESHOLD:
                    return -np.inf
                if not math.isfinite(fv) and fv < 0.0:
                    return -np.inf
                return fv

            ap_depth_d0 = int(d.get("ap_depth_d0", 0))
            ap_depth_d1 = int(d.get("ap_depth_d1", 0))
            # Defensive: enforce the invariant ``depth > 0`` iff
            # ``pending_ap >= 0``.  A malformed state JSON with
            # inconsistent values would otherwise confuse the kernel.
            if float(d.get("pending_ap_d0", -1.0)) < 0.0:
                ap_depth_d0 = 0
            if float(d.get("pending_ap_d1", -1.0)) < 0.0:
                ap_depth_d1 = 0

            new_state = _DetectorState(
                last_abs_fire_time_ns_d0=_restore_fire_time(d.get("last_fire_d0", -np.inf)),
                last_abs_fire_time_ns_d1=_restore_fire_time(d.get("last_fire_d1", -np.inf)),
                pending_ap_time_d0=float(d.get("pending_ap_d0", -1.0)),
                pending_ap_time_d1=float(d.get("pending_ap_d1", -1.0)),
                ap_depth_d0=ap_depth_d0,
                ap_depth_d1=ap_depth_d1,
                next_dc_time_d0=float(next_dc_d0),
                next_dc_time_d1=float(next_dc_d1),
                carry_over_events=carry_over,
                total_time_processed_ns=float(d.get("total_time_processed_ns", 0.0)),
                afterpulse_total_count=int(d.get("afterpulse_total_count", 0)),
                last_path=d.get("last_path", None),
            )

            for label, lf in (
                ("last_fire_d0", new_state.last_abs_fire_time_ns_d0),
                ("last_fire_d1", new_state.last_abs_fire_time_ns_d1),
            ):
                if not (math.isinf(lf) and lf < 0.0) and not (
                    math.isfinite(lf) and lf >= 0.0
                ):
                    raise ConfigurationError(
                        f"State field {label!r} = {lf!r} is invalid: must be "
                        f"-inf (never fired) or a non-negative finite value "
                        f"(review-v5 F-19)."
                    )
            self._internal_state = new_state
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to load state: {e}") from e

    def clone(self) -> "SinglePhotonDetector":
        """Deep-copy this detector including runtime state.

        Uses :func:`dataclasses.replace` to create a new config-level
        copy, then manually constructs an independent
        :class:`_DetectorState` so that mutable runtime state (fire
        times, pending APs, dark-count scheduling, etc.) is not shared
        between the original and the clone.  The previous
        :func:`copy.deepcopy` approach was replaced by this explicit
        construction to avoid deep-copying nested numpy arrays and
        JSON-fragile edge cases (review-v17 H3 / review F51/PF4).
        """
        # Shallow-copy the config + state container.  ``replace`` with
        # no field overrides creates a new instance that shares the
        # mutable ``_internal_state`` reference; we then explicitly
        # replace it with a fresh copy.
        new = replace(self)
        new._internal_state = _DetectorState(
            last_abs_fire_time_ns_d0=self._internal_state.last_abs_fire_time_ns_d0,
            last_abs_fire_time_ns_d1=self._internal_state.last_abs_fire_time_ns_d1,
            pending_ap_time_d0=self._internal_state.pending_ap_time_d0,
            pending_ap_time_d1=self._internal_state.pending_ap_time_d1,
            ap_depth_d0=self._internal_state.ap_depth_d0,
            ap_depth_d1=self._internal_state.ap_depth_d1,
            next_dc_time_d0=self._internal_state.next_dc_time_d0,
            next_dc_time_d1=self._internal_state.next_dc_time_d1,
            carry_over_events=tuple(self._internal_state.carry_over_events),
            total_time_processed_ns=self._internal_state.total_time_processed_ns,
            afterpulse_total_count=self._internal_state.afterpulse_total_count,
            # Review-v8 R9-21: copy ``last_path`` so the clone has
            # independent (but identically-initialized) path tracking.
            last_path=self._internal_state.last_path,
        )
        return new

    _NOISE_FIELDS: ClassVar[Tuple[str, ...]] = ()

    def noiseless_copy(self) -> "SinglePhotonDetector":
        """Return a fresh detector with the same configuration but all
        noise-related fields zeroed out.  Detector efficiency and the
        detector type are preserved (the result is "ideal *detection*
        of an ideal channel" -- not a perfect-efficiency detector).

        The returned detector starts in a clean initial state.
        """
        kwargs: Dict[str, Any] = {name: 0.0 for name in self._NOISE_FIELDS}
        new = replace(self, **kwargs)
        new._internal_state = new._create_initial_state()
        return new

    # ---- Validation ----

    def _validate_params(self):
        # Accept either a DoubleClickPolicy enum or the ROGERS_2007 sentinel.
        if not (
            isinstance(self.double_click_policy, DoubleClickPolicy)
            or _is_rogers_policy(self.double_click_policy)
        ):
            raise ParameterValidationError(
                "Parameter 'double_click_policy' must be a DoubleClickPolicy Enum "
                "or the module-level ROGERS_2007_POLICY sentinel."
            )
        if not isinstance(self.detector_type, DetectorType):
            raise ParameterValidationError(
                "Parameter 'detector_type' must be a DetectorType Enum."
            )
        if not isinstance(self.dead_time_model, DeadTimeModel):
            raise ParameterValidationError(
                "Parameter 'dead_time_model' must be a DeadTimeModel Enum."
            )
        if not isinstance(self.afterpulse_model, AfterpulseModel):
            raise ParameterValidationError(
                "Parameter 'afterpulse_model' must be an AfterpulseModel Enum."
            )

        # Rogers et al. (2007) §3/§6: individual SPADs are non-paralyzable.
        if self.dead_time_model == DeadTimeModel.PARALYZABLE:
            if self.strict_mode:
                raise ParameterValidationError(
                    "DeadTimeModel.PARALYZABLE is not supported: per Rogers "
                    "et al. (2007), individual SPADs are non-paralyzable. "
                    "For Rogers-style paralyzable basis-level behavior, use "
                    "DeadTimeModel.NON_PARALYZABLE with the ROGERS_2007_POLICY "
                    "double-click policy.  To suppress this error, set "
                    "strict_mode=False (not recommended for research use)."
                )
            logger.warning(
                "DeadTimeModel.PARALYZABLE is deprecated: per Rogers et al. "
                "(2007), individual SPADs are non-paralyzable.  Treating as "
                "NON_PARALYZABLE.  For Rogers-style paralyzable basis-level "
                "behavior, use DoubleClickPolicy ROGERS_2007_POLICY instead."
            )

        prob_params = [
            "det_eff_d0", "det_eff_d1",
            "qber_intrinsic", "misalignment", "afterpulse_prob",
        ]
        for name in prob_params:
            value = getattr(self, name)
            if not is_valid_probability(value):
                raise ParameterValidationError(
                    f"Parameter '{name}' must be a valid probability in [0, 1]."
                )

        nonneg_params = [
            "dark_rate", "dead_time_ns", "jitter_fwhm_ns",
            "bias_voltage", "breakdown_voltage", "ref_bias_voltage",
            "gate_width_ns",
        ]
        for name in nonneg_params:
            value = getattr(self, name)
            if not is_finite_non_negative(value):
                raise ParameterValidationError(
                    f"Parameter '{name}' must be finite and non-negative."
                )

        # Per-detector jitter (review P8).
        for name in ("jitter_fwhm_ns_d0", "jitter_fwhm_ns_d1"):
            value = getattr(self, name)
            if value is not None and not is_finite_non_negative(value):
                raise ParameterValidationError(
                    f"Parameter '{name}' must be None or finite and non-negative."
                )

        # Review-v17 H2: validate per-detector breakdown voltage overrides.
        for name in ("breakdown_voltage_d0", "breakdown_voltage_d1"):
            value = getattr(self, name)
            if value is not None and not is_finite_non_negative(value):
                raise ParameterValidationError(
                    f"Parameter '{name}' must be None or finite and "
                    f"non-negative (review-v17 H2)."
                )

        # Review-v19 F-01: validate breakdown temperature coefficient.
        if not math.isfinite(self.breakdown_temp_coeff_mv_per_k):
            raise ParameterValidationError(
                f"Parameter 'breakdown_temp_coeff_mv_per_k' must be "
                f"finite; got {self.breakdown_temp_coeff_mv_per_k!r}."
            )

        # Geiger-mode guard: SPAD-only.  For SNSPD/PNRD this check is
        # skipped because the bias/breakdown semantics do not apply.
        # Review-v12 F8: also check per-detector bias_voltage overrides.
        # The previous code only checked the shared bias_voltage, which
        # allowed a user to set bias_voltage_d0 or bias_voltage_d1 below
        # breakdown while the shared value was above, silently accepting
        # an unphysical SPAD configuration.
        if self.detector_type == DetectorType.SPD:
            if self.bias_voltage <= self.breakdown_voltage:
                raise ParameterValidationError(
                    f"SPAD must operate in Geiger mode: bias_voltage "
                    f"({self.bias_voltage}) must strictly exceed breakdown_voltage "
                    f"({self.breakdown_voltage}).",
                    param_name="bias_voltage",
                    param_value=self.bias_voltage,
                )
            # Per-detector overrides: if supplied, each must also
            # strictly exceed its corresponding breakdown_voltage.
            # Review-v17 H2: use per-detector breakdown_voltage_d0/d1
            # when available, falling back to the shared value.
            bv_d0 = self.breakdown_voltage_d0 if self.breakdown_voltage_d0 is not None else self.breakdown_voltage
            bv_d1 = self.breakdown_voltage_d1 if self.breakdown_voltage_d1 is not None else self.breakdown_voltage
            if self.bias_voltage_d0 is not None and self.bias_voltage_d0 <= bv_d0:
                raise ParameterValidationError(
                    f"SPAD per-detector bias_voltage_d0 "
                    f"({self.bias_voltage_d0}) must strictly exceed "
                    f"breakdown_voltage_d0 ({bv_d0}) "
                    f"(review-v12 F8 / review-v17 H2).  A SPAD cannot operate below "
                    f"Geiger threshold on any detector channel.",
                    param_name="bias_voltage_d0",
                    param_value=self.bias_voltage_d0,
                )
            if self.bias_voltage_d1 is not None and self.bias_voltage_d1 <= bv_d1:
                raise ParameterValidationError(
                    f"SPAD per-detector bias_voltage_d1 "
                    f"({self.bias_voltage_d1}) must strictly exceed "
                    f"breakdown_voltage_d1 ({bv_d1}) "
                    f"(review-v12 F8 / review-v17 H2).  A SPAD cannot operate below "
                    f"Geiger threshold on any detector channel.",
                    param_name="bias_voltage_d1",
                    param_value=self.bias_voltage_d1,
                )

        # SNSPD bias-current sanity check (review P3).
        if self.detector_type == DetectorType.SNSPD:
            if self.bias_current is not None and self.switching_current is not None:
                if self.switching_current <= 0.0:
                    raise ParameterValidationError(
                        "Parameter 'switching_current' must be strictly positive."
                    )
                if self.bias_current <= 0.0 or self.bias_current >= self.switching_current:
                    raise ParameterValidationError(
                        f"SNSPD bias_current ({self.bias_current}) must lie in "
                        f"(0, switching_current={self.switching_current})."
                    )

        # SNSPD afterpulse sanity check (review P11).
        if self.detector_type == DetectorType.SNSPD and self.strict_mode:
            if self.afterpulse_prob > 0.01:
                raise ParameterValidationError(
                    f"SNSPD afterpulse_prob ({self.afterpulse_prob}) exceeds 0.01. "
                    f"Real SNSPDs have afterpulse < 1e-5; the supplied value is "
                    f"physically unrealistic.  Set strict_mode=False to override."
                )
            if self.temperature_k > 100.0 or self.ref_temperature_k > 100.0:
                raise ParameterValidationError(
                    f"SNSPD temperature_k={self.temperature_k}K "
                    f"(ref={self.ref_temperature_k}K) is above 100K -- "
                    f"SNSPDs are not operated at these temperatures "
                    f"(review-v5 F-17).  Either lower the temperature, "
                    f"set strict_mode=False, or supply a measured dark "
                    f"rate directly."
                )

        if self.dark_rate_d1 is not None and not is_finite_non_negative(self.dark_rate_d1):
            raise ParameterValidationError(
                "Parameter 'dark_rate_d1' must be finite and non-negative when provided."
            )

        for name in ("det_eff_d0_z", "det_eff_d0_x", "det_eff_d1_z", "det_eff_d1_x"):
            value = getattr(self, name)
            if value is not None and not is_valid_probability(value):
                raise ParameterValidationError(
                    f"Parameter '{name}' must be None or a valid probability in [0, 1]."
                )

        if not math.isfinite(self.afterpulse_lifetime_ns) or self.afterpulse_lifetime_ns < 0.0:
            raise ParameterValidationError(
                "Parameter 'afterpulse_lifetime_ns' must be finite and non-negative."
            )

        if not math.isfinite(self.temperature_k) or self.temperature_k <= 0.0:
            raise ParameterValidationError(
                "Parameter 'temperature_k' must be finite and positive."
            )

        if not math.isfinite(self.ref_temperature_k) or self.ref_temperature_k <= 0.0:
            raise ParameterValidationError(
                "Parameter 'ref_temperature_k' must be finite and positive."
            )

        if self.gated and self.gate_width_ns <= 0.0:
            raise ParameterValidationError(
                "Parameter 'gate_width_ns' must be strictly positive when gated=True."
            )

        if self.gated:
            if not math.isfinite(self.gate_offset_ns) or self.gate_offset_ns < 0.0:
                raise ParameterValidationError(
                    "Parameter 'gate_offset_ns' must be finite and non-negative "
                    "when gated=True."
                )

        if self.detector_type == DetectorType.PNRD and self.gated:
            raise ParameterValidationError(
                "Gated mode is not supported in the PNRD path: the per-pulse "
                "Poisson dark-count model over-counts dark counts when the "
                "gate window is narrower than the pulse period (review-v5 "
                "F-34).  Use the threshold path (SPD/SNSPD) for gated "
                "detection."
            )

        # PNRD + stateful effects: enforce in strict mode; warn in
        # non-strict mode.  F-M10: silently ignoring these parameters
        # can mislead callers who expect them to take effect.
        if self.detector_type == DetectorType.PNRD:
            pnrd_ignored = []
            if self.dead_time_ns > 0.0:
                if self.strict_mode:
                    raise ParameterValidationError(
                        "Strict mode: dead_time_ns is not currently modeled in the PNRD path."
                    )
                pnrd_ignored.append("dead_time_ns")
            if self.afterpulse_prob > 0.0:
                if self.strict_mode:
                    raise ParameterValidationError(
                        "Strict mode: afterpulse_prob is not currently modeled in the PNRD path."
                    )
                pnrd_ignored.append("afterpulse_prob")
            if self.jitter_fwhm_ns > 0.0:
                if self.strict_mode:
                    raise ParameterValidationError(
                        "Strict mode: jitter_fwhm_ns is not currently modeled in the PNRD path."
                    )
                pnrd_ignored.append("jitter_fwhm_ns")
            if pnrd_ignored:
                logger.warning(
                    "PNRD path does not model the following parameters, which "
                    "will be silently ignored: %s.  Use the threshold path "
                    "(SPD/SNSPD) if you need these effects (F-M10).",
                    ", ".join(pnrd_ignored),
                )

            if self.strict_mode and (
                self.det_eff_d0_z is not None or self.det_eff_d0_x is not None
                    or self.det_eff_d1_z is not None or self.det_eff_d1_x is not None
            ):
                raise ParameterValidationError(
                    "Strict mode: basis-dependent detector efficiency "
                    "(det_eff_d{0,1}_{z,x}) is not currently modeled in the "
                    "PNRD path (review-v5 F-14).  Either switch to the "
                    "threshold path (SPD/SNSPD) or set the basis-specific "
                    "fields to None and use the scalar det_eff_d{0,1} values."
                )

        if not isinstance(self.ap_reschedule_release, APRescheduleMode):
            raise ParameterValidationError(
                f"Parameter 'ap_reschedule_release' must be an "
                f"APRescheduleMode enum member; got "
                f"{self.ap_reschedule_release!r}.  Legacy "
                f"string values 'end' and 'uniform' are accepted at "
                f"construction time and coerced; typos raise."
            )

        if (
            self.detector_type == DetectorType.PNRD
            and self.detector_topology == DetectorTopology.SHARED_SPAD
        ):
            msg = (
                "PNRD path does not model SHARED_SPAD topology: the "
                "PNRD path has no dead time, but a shared-SPAD receiver "
                "requires a shared dead-time clock.  Use "
                "INDEPENDENT_SPADS for PNRD, or switch to the threshold "
                "path (SPD) with SHARED_SPAD for a shared-dead-time "
                "threshold detector."
            )
            if self.strict_mode:
                raise ParameterValidationError(msg)
            logger.warning(msg)

        if (
            self.detector_type == DetectorType.SNSPD
            and self.strict_mode
            and not self.allow_qualitative_snspd_scaler
        ):
            pinned = is_close(self.temperature_k, self.ref_temperature_k) and is_close(
                self.bias_voltage, self.ref_bias_voltage
            )
            if not pinned:
                raise ParameterValidationError(
                    f"Strict mode: SNSPD dark-rate scaler is documented "
                    f"as QUALITATIVE ONLY (F-12).  The operating point "
                    f"is not pinned: temperature_k="
                    f"{self.temperature_k} (ref={self.ref_temperature_k}), "
                    f"bias_voltage={self.bias_voltage} (ref="
                    f"{self.ref_bias_voltage}).  For publication-bound "
                    f"SNSPD work, callers MUST supply measured dark "
                    f"rates at the operating point (set ``dark_rate`` "
                    f"to the measured value and pin "
                    f"``ref_temperature_k = temperature_k`` and "
                    f"``ref_bias_voltage = bias_voltage`` so the scaler "
                    f"evaluates to 1.0).  Alternatively, set "
                    f"``allow_qualitative_snspd_scaler=True`` to "
                    f"explicitly acknowledge that the qualitative "
                    f"scaler is being used (NOT recommended for "
                    f"publication-bound work)."
                )

        if self.recovery_time_ns > 0.0:
            msg = (
                f"recovery_time_ns={self.recovery_time_ns} is set, but "
                f"the current kernel uses a HARD cut-off recovery model "
                f"(F-13).  Gradual recovery (exponential or linear ramp) "
                f"is a future-work item; the value is currently ignored. "
                f"Real SPADs recover gradually; the hard-cutoff "
                f"approximation can bias high-flux gain estimates.  Set "
                f"recovery_time_ns=0 to silence this message, or set "
                f"strict_mode=False to suppress the error."
            )
            if self.strict_mode:
                raise NotImplementedError(msg)
            logger.warning(msg)
        elif self.recovery_time_ns < 0.0 or not math.isfinite(self.recovery_time_ns):
            raise ParameterValidationError(
                f"recovery_time_ns must be finite and non-negative; got "
                f"{self.recovery_time_ns!r} (F-13)."
            )

        if self.max_count_rate_hz is not None:
            if (
                not math.isfinite(self.max_count_rate_hz)
                or self.max_count_rate_hz <= 0.0
            ):
                raise ParameterValidationError(
                    f"max_count_rate_hz must be a positive finite number or "
                    f"None; got {self.max_count_rate_hz!r} (F-14)."
                )

        for name in (
            "temperature_k_d0", "temperature_k_d1",
            "bias_voltage_d0", "bias_voltage_d1",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if name.startswith("temperature_k"):
                if not math.isfinite(value) or value <= 0.0:
                    raise ParameterValidationError(
                        f"Parameter '{name}' must be finite and positive when "
                        f"provided (F-10)."
                    )
            else:  # bias_voltage_d{0,1}
                if not math.isfinite(value) or value < 0.0:
                    raise ParameterValidationError(
                        f"Parameter '{name}' must be finite and non-negative "
                        f"when provided (F-10)."
                    )

        if self.afterpulse_prob > 0.0 and self.afterpulse_lifetime_ns > 0.0:
            # Use dark_rate as a lower-bound fire rate.  Actual fire
            # rate may be much higher when signal flux is non-trivial.
            r_fire_lower = max(self.dark_rate, 1.0)
            occupancy = (
                self.afterpulse_prob
                * r_fire_lower
                * self.afterpulse_lifetime_ns
                * 1e-9
            )
            if occupancy > 0.1:
                msg = (
                    f"Single-pending-AP approximation may under-count "
                    f"afterpulses (F-07): estimated occupancy = "
                    f"p_ap * r_fire * tau_ap = {occupancy:.4g} > 0.1 "
                    f"(p_ap={self.afterpulse_prob}, r_fire_lower="
                    f"{r_fire_lower} Hz, tau_ap="
                    f"{self.afterpulse_lifetime_ns} ns).  Real SPADs "
                    f"have many independent trap levels decaying in "
                    f"parallel; the current kernel models only "
                    f"{self.max_pending_aps_per_det} pending AP per "
                    f"detector.  Consider reducing afterpulse_prob, "
                    f"using a shorter afterpulse_lifetime_ns, or "
                    f"documenting this as a quantitative validity "
                    f"limitation in any publication."
                )
                if self.strict_mode:
                    raise ParameterValidationError(msg)
                logger.warning(msg)

        for f in dataclass_fields(self):
            if f.name == "_internal_state":
                continue
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if value is None or isinstance(value, (bool, int, float, str, Enum)):
                continue
            raise ParameterValidationError(
                f"Configuration field '{f.name}' has non-JSON-serializable "
                f"type {type(value).__name__}; expected bool, int, float, "
                f"str, Enum, or None.  Numpy arrays and other objects are "
                f"rejected because their ``str()`` representation is "
                f"Python-version dependent (review-v4 F-22)."
            )

    def _create_initial_state(self) -> _DetectorState:
        return _DetectorState()

    def _config_hash(self) -> str:
        """Stable hash of the configuration.

        Returns the full 64-character SHA-256 hex digest to avoid
        collision risk in large config spaces (F-L1).
        """
        config_dict = self.to_config_dict(include_experimental=True)
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()

    # ---- Physical helpers ----

    def _calculate_dynamic_dark_rates(self) -> Tuple[float, float]:
        """Return ``(dark_rate_d0_hz, dark_rate_d1_hz)`` after applying
        detector-type-aware temperature/bias scaling.

        - For ``SPD``: classic SPAD trap-assisted scaling with activation
          energy ``SPAD_DARK_ACTIVATION_EV`` and quadratic excess-bias
          factor.  This is a crude approximation (review F37 / F-24);
          real SPAD dark rate scales with overvoltage via trap-assisted
          tunneling (exponential).  The quadratic form is retained for
          backward compatibility; callers requiring accuracy should
          supply measured dark rates at the operating point.
        - For ``SNSPD``: Planck-spectrum black-body photon-flux scaling
          (review P2) plus bias-current scaling (review P3).  Bias
          voltage is not used.  The scaler is documented as
          QUALITATIVE ONLY (F-12); strict mode requires the caller to
          pin the operating point.
        - For ``PNRD``: uses the SPAD scaling by default (PNRD arrays
          are typically InGaAs SPADs).

        **Review-v18 F-03 (HIGH): Caching.**  The result is cached for
        the duration of a single ``simulate_detection`` call to avoid
        redundant Planck integral recomputation (SNSPD) and Arrhenius
        calculation (SPAD).  The cache is invalidated at the start of
        each ``simulate_detection`` call.
        """
        # Review-v18 F-03: check per-call cache.
        cached = getattr(self, "_cached_dynamic_dark_rates", None)
        if cached is not None:
            return cached
        base_rate_0 = self.dark_rate
        base_rate_1 = self.dark_rate_d1 if self.dark_rate_d1 is not None else self.dark_rate

        if is_close(base_rate_0, 0.0) and is_close(base_rate_1, 0.0):
            self._cached_dynamic_dark_rates = (0.0, 0.0)
            return 0.0, 0.0

        def _scaler_for(
            temp_override: Optional[float],
            bias_override: Optional[float],
            breakdown_override: Optional[float],
        ) -> float:
            if temp_override is None and bias_override is None:
                # Symmetric path: use the shared scaler.
                if self.detector_type == DetectorType.SNSPD:
                    return self._snspd_dark_scaler()
                return self._spad_dark_scaler()
            # Asymmetric path: compute the scaler with the per-detector
            # operating point substituted into the shared fields.  We
            # construct a lightweight namespace that has the same
            # attributes the scaler methods read (``temperature_k``,
            # ``bias_voltage``, ``breakdown_voltage``, ``ref_*``,
            # ``bias_current``, ``switching_current``, ``strict_mode``,
            # ``allow_qualitative_snspd_scaler``) and invoke the scaler
            # method as an unbound function on it.  This bypasses the
            # dataclass ``__post_init__`` validation (e.g. Geiger-mode
            # guard) because the override values are not necessarily
            # valid detector configurations on their own.
            # Review-v17 H2/N2: use per-detector breakdown_voltage
            # overrides when computing the dark-rate scaler for an
            # asymmetric detector.  The previous code used the shared
            # ``breakdown_voltage`` even when the bias override was
            # per-detector, making the excess-bias calculation wrong.
            # Now the caller passes the correct per-detector breakdown
            # override explicitly.
            from types import SimpleNamespace
            effective_breakdown = (
                breakdown_override if breakdown_override is not None
                else self.breakdown_voltage
            )
            stand_in = SimpleNamespace(
                temperature_k=(
                    float(temp_override) if temp_override is not None
                    else self.temperature_k
                ),
                bias_voltage=(
                    float(bias_override) if bias_override is not None
                    else self.bias_voltage
                ),
                breakdown_voltage=effective_breakdown,
                ref_temperature_k=self.ref_temperature_k,
                ref_bias_voltage=self.ref_bias_voltage,
                bias_current=self.bias_current,
                switching_current=self.switching_current,
                strict_mode=False,  # do not raise inside the stand-in
                allow_qualitative_snspd_scaler=True,
                # Review-v19 F-01: pass breakdown_temp_coeff to stand-in
                # so _spad_dark_scaler can use it.
                breakdown_temp_coeff_mv_per_k=self.breakdown_temp_coeff_mv_per_k,
            )
            if self.detector_type == DetectorType.SNSPD:
                return SinglePhotonDetector._snspd_dark_scaler(stand_in)
            return SinglePhotonDetector._spad_dark_scaler(stand_in)

        scaler_0 = _scaler_for(self.temperature_k_d0, self.bias_voltage_d0, self.breakdown_voltage_d0)
        scaler_1 = _scaler_for(self.temperature_k_d1, self.bias_voltage_d1, self.breakdown_voltage_d1)

        if self.strict_mode and not self.allow_qualitative_snspd_scaler:
            for label, scaler in (("D0", scaler_0), ("D1", scaler_1)):
                dev = abs(scaler - 1.0)
                if dev > 0.05:
                    raise ParameterValidationError(
                        f"{self.detector_type.name} dark-rate scaler "
                        f"({label}) deviates from 1.0 by {dev:.4g} "
                        f"(scaler={scaler:.6g}).  The built-in SPAD/SNSPD "
                        f"scaler is documented as CRUDE / QUALITATIVE "
                        f"ONLY (F-12 / review-v7 R7-04/R7-05); for "
                        f"publication-bound work callers MUST supply "
                        f"measured dark rates at the operating point.  "
                        f"Set ``ref_temperature_k = temperature_k`` and "
                        f"``ref_bias_voltage = bias_voltage`` (and the "
                        f"per-detector equivalents when overrides are "
                        f"used) so the scaler evaluates to 1.0, OR set "
                        f"``allow_qualitative_snspd_scaler=True`` to "
                        f"explicitly acknowledge the qualitative scaler "
                        f"is being used, OR set ``strict_mode=False`` "
                        f"to silence this error (NOT recommended for "
                        f"publication-bound work)."
                    )

        scaler_0 = max(0.0, scaler_0)
        scaler_1 = max(0.0, scaler_1)
        result = base_rate_0 * scaler_0, base_rate_1 * scaler_1
        # Review-v18 F-03: store per-call cache.
        self._cached_dynamic_dark_rates = result
        return result

    def _spad_dark_scaler(self) -> float:
        """SPAD trap-assisted dark-rate scaling factor.

        Combines a temperature factor (Arrhenius with activation energy
        ``_SPAD_DARK_ACTIVATION_EV``) and a voltage factor (quadratic in
        excess bias).  Both are approximations.

        **Review-v19 F-01 fix (Critical):** The operating-point breakdown
        voltage is now adjusted by the temperature coefficient
        ``breakdown_temp_coeff_mv_per_k`` (default 50 mV/K for InGaAs):
        ``bdv_op = breakdown_voltage + coeff * 1e-3 * (temperature_k -
        ref_temperature_k)``.  The reference-point breakdown voltage
        remains ``breakdown_voltage`` (assumed to be measured at
        ``ref_temperature_k``).  This corrects the previous bug where
        the same ``breakdown_voltage`` was used for BOTH the operating
        and reference excess-bias, silently producing wrong values when
        ``ref_temperature_k != temperature_k``.  Set
        ``breakdown_temp_coeff_mv_per_k = 0.0`` to disable the correction
        and revert to the old behavior.  For publication-bound work,
        callers should still supply measured dark rates or pin the
        operating point (``ref_temperature_k = temperature_k``).
        """
        # Review-v19 F-01: use temperature-dependent breakdown voltage
        # at the operating point.  The reference breakdown voltage is
        # assumed to be measured at ref_temperature_k and stays at
        # self.breakdown_voltage.  The operating breakdown voltage shifts
        # by coeff * delta_T.
        coeff_V_per_K = self.breakdown_temp_coeff_mv_per_k * 1e-3  # mV/K -> V/K
        delta_T = self.temperature_k - self.ref_temperature_k
        bdv_op = self.breakdown_voltage + coeff_V_per_K * delta_T
        excess_bias = self.bias_voltage - bdv_op
        ref_excess_bias = self.ref_bias_voltage - self.breakdown_voltage
        if ref_excess_bias <= EPS:
            ref_excess_bias = _REF_EXCESS_BIAS_FALLBACK_V
        voltage_factor = (excess_bias / ref_excess_bias) ** 2

        kT = CONST_BOLTZMANN * self.temperature_k
        ref_kT = CONST_BOLTZMANN * self.ref_temperature_k
        e_act = _SPAD_DARK_ACTIVATION_EV * _EV_TO_JOULE

        exponent = (e_act / ref_kT) - (e_act / kT)
        if exponent < -700.0 or exponent > 700.0:
            msg = (
                f"SPAD dark-rate Arrhenius exponent ({exponent:.4g}) "
                f"is outside the safe numerical range [-700, 700] "
                f"(review-v8 R9-24).  This indicates an extreme-"
                f"temperature configuration: temperature_k="
                f"{self.temperature_k}K, ref_temperature_k="
                f"{self.ref_temperature_k}K, activation_energy="
                f"{_SPAD_DARK_ACTIVATION_EV}eV.  The SPAD dark-rate "
                f"scaler is documented as CRUDE / QUALITATIVE ONLY; "
                f"for publication-bound work callers MUST supply "
                f"measured dark rates at the operating point.  Set "
                f"ref_temperature_k=temperature_k so the scaler "
                f"evaluates to 1.0, OR set strict_mode=False to "
                f"silence this error (NOT recommended for "
                f"publication-bound work)."
            )
            if self.strict_mode:
                raise ParameterValidationError(msg)
            logger.warning(msg)
            exponent = max(-700.0, min(700.0, exponent))
        temp_factor = math.exp(exponent)
        return temp_factor * voltage_factor

    def _snspd_dark_scaler(self) -> float:
        """SNSPD dark-rate scaling factor.

        ``x`` well below the clamp threshold (typically ``x <= 0.95``).
        """
        temp_factor = self._snspd_blackbody_temp_factor()

        bias_factor = 1.0
        if self.bias_current is not None and self.switching_current is not None:
            x = clamp_probability(self.bias_current / self.switching_current)
            # Empirical bias-current factor: grows from 0 to 1 as x -> 1.
            # alpha = 0.5 is a fitting constant; for quantitative work,
            # replace with a measured scaling law.
            alpha = _SNSPD_BIAS_FACTOR_ALPHA
            if x >= 1.0 - EPS:
                bias_factor = 1.0
            elif x <= 0.0:
                bias_factor = 0.0
            else:
                bias_factor = math.exp(-alpha / (1.0 - x))
                # Normalize so that x=0.9 gives ~0.5 (arbitrary reference).
                # This is qualitative; do not use for publication-bound results.
                ref = math.exp(-alpha / (1.0 - _SNSPD_BIAS_FACTOR_REF_X))
                bias_factor = bias_factor / ref if ref > 0 else 1.0
                if bias_factor > _SNSPD_BIAS_FACTOR_CLAMP:
                    msg = (
                        f"SNSPD bias-current scaling factor "
                        f"({bias_factor:.4g}) exceeds the qualitative "
                        f"clamp at {_SNSPD_BIAS_FACTOR_CLAMP} "
                        f"(F-26 / review-v8 R9-20).  This "
                        f"indicates bias_current="
                        f"{self.bias_current}uA is very close to "
                        f"switching_current={self.switching_current}uA "
                        f"(ratio x={x:.6g}); the empirical "
                        f"f(x)=exp(-alpha/(1-x)) diverges as x -> 1.  "
                        f"The SNSPD dark-rate scaler is documented as "
                        f"QUALITATIVE ONLY; for publication-bound work "
                        f"supply a measured dark rate directly."
                    )
                    if self.strict_mode:
                        raise ParameterValidationError(msg)
                    logger.warning(msg)
                    bias_factor = _SNSPD_BIAS_FACTOR_CLAMP
                elif bias_factor < 0.0:
                    # Defensive: ``math.exp`` always returns >= 0,
                    # so this branch is unreachable in normal use.
                    # Retained for robustness against future code
                    # changes that might produce a negative factor.
                    bias_factor = 0.0

        return temp_factor * bias_factor

    def _snspd_blackbody_temp_factor(self) -> float:
        """Black-body photon-flux temperature scaling for SNSPD.

        Computes ``Phi(T) / Phi(T_ref)`` where ``Phi`` is the photon
        flux per unit area per unit wavelength above the cutoff
        wavelength ``lambda_cut`` (set to the SNSPD detection cutoff,
        ~1550 nm by default).

        Raises :class:`ParameterValidationError` in strict mode if
        ``T > 100 K`` and the caller has not explicitly opted in via
        ``strict_mode = False``.  SNSPDs are typically operated at
        2-4 K, so this limitation is unlikely to be hit in practice.
        """
        try:
            h = CONST_PLANCK          # Planck constant, J*s (R7-21)
            c = CONST_SPEED_OF_LIGHT  # speed of light, m/s (R7-21)
            k = CONST_BOLTZMANN       # Boltzmann constant, J/K
            lam_cut = _SNSPD_REF_WAVELENGTH_NM * 1e-9  # m
            nu_cut = c / lam_cut

            x_t = h * nu_cut / (k * self.temperature_k)
            x_ref = h * nu_cut / (k * self.ref_temperature_k)

            if self.strict_mode and (
                self.temperature_k > 100.0 or self.ref_temperature_k > 100.0
            ):
                raise ParameterValidationError(
                    f"SNSPD black-body scaling requested at T="
                    f"{self.temperature_k}K (ref={self.ref_temperature_k}K), "
                    f"which is above 100K -- SNSPDs are not operated at "
                    f"these temperatures (review-v5 F-17).  Either lower "
                    f"the temperature, set strict_mode=False, or supply "
                    f"a measured dark rate directly."
                )

            def _phi(x: float, T: float) -> float:
                if x > 700.0:
                    return 0.0
                prefactor = 2.0 * (k * T) ** 3 / (c * c * h ** 3)
                total = 0.0
                e_neg_x = math.exp(-x)
                if e_neg_x <= 0.0:
                    return 0.0
                # n=1 term: e^{-x} (x^2 + 2x + 2)
                total = e_neg_x * (x * x + 2.0 * x + 2.0)
                # n>=2 terms: e^{-n x} (x^2/n + 2x/n^2 + 2/n^3)
                # Review-v15 F-35: the previous convergence check
                # ``if e_neg_x < 0.999`` skipped the series when
                # e^{-x} was close to 1 (i.e., x close to 0), where
                # each series term is ~1/n^3 and the series DOES
                # converge (sum is finite), but converges slowly.  For
                # x < 0.001, the n=1 term alone under-estimates the
                # black-body flux by ~20%.  The fix: always iterate up
                # to MAX_TERMS, relying on the relative-tolerance break
                # to stop early when the series converges quickly.  Only
                # skip iteration when e_neg_x is truly zero (overflow)
                # or when x is exactly zero (handled above).
                if e_neg_x > 1e-300:
                    e_neg_x_n = e_neg_x  # e^{-n x} for n=1
                    n = 2
                    while n <= _SNSPD_PLANCK_MAX_TERMS:
                        e_neg_x_n *= e_neg_x  # now e^{-n x}
                        if e_neg_x_n < 1e-300:
                            break
                        term = e_neg_x_n * (
                            x * x / n + 2.0 * x / (n * n) + 2.0 / (n ** 3)
                        )
                        total += term
                        if term < _SNSPD_PLANCK_TERM_REL_TOL * total:
                            break
                        n += 1
                return prefactor * total

            phi_t = _phi(x_t, self.temperature_k)
            phi_ref = _phi(x_ref, self.ref_temperature_k)
            if phi_ref <= 0.0:
                return 1.0
            return phi_t / phi_ref
        except (OverflowError, ValueError, ZeroDivisionError):
            # Review-v17 M13: the Arrhenius fallback with hardcoded
            # e_act = 0.8 eV is qualitative-only and produces
            # significantly different scaling than the Planck model,
            # particularly at T < 10 K.  In strict mode, raise instead
            # of falling back so that publication-bound work cannot
            # silently use the wrong scaling.
            if self.strict_mode:
                raise ParameterValidationError(
                    "SNSPD black-body Planck integral failed in strict "
                    "mode; cannot fall back to qualitative Arrhenius "
                    "approximation (review-v17 M13).  Either pin the "
                    "operating point (set ref_temperature_k=temperature_k "
                    "so the scaler evaluates to 1.0) or set "
                    "strict_mode=False to allow the Arrhenius fallback "
                    "(NOT recommended for publication-bound work)."
                )
            logger.warning(
                "SNSPD black-body scaling failed; falling back to Arrhenius.",
                exc_info=True,
            )
            kT = CONST_BOLTZMANN * self.temperature_k
            ref_kT = CONST_BOLTZMANN * self.ref_temperature_k
            e_act = 0.8 * _EV_TO_JOULE  # legacy activation energy
            exponent = (e_act / ref_kT) - (e_act / kT)
            exponent = max(-700.0, min(700.0, exponent))
            return math.exp(exponent)

    def _calculate_imd_intensity(
        self,
        channel_transmittance: float,
        rng: np.random.Generator,
        num_samples: int,
        *,
        det_eff_d0: Optional[Union[float, NDArray[Any]]] = None,
        det_eff_d1: Optional[Union[float, NDArray[Any]]] = None,
        **kwargs,
    ) -> Tuple[float, float]:
        """Return per-pulse mean IMD/FWM photon-number contributions for
        D0 and D1.
        """
        del rng  # reserved for future stochastic IMD models (review F57)

        mu_imd_base_0 = 0.0
        mu_imd_base_1 = 0.0
        n_channels = int(kwargs.get("N_channels", 1))
        modulation_index = float(kwargs.get("modulation_index", 0.0))
        mu_signal = float(kwargs.get("mu_signal", 0.0))

        def _mean_eff(eff_arg: Optional[Union[float, NDArray[Any]]], fallback: float) -> float:
            if eff_arg is None:
                return float(fallback)
            arr = np.asarray(eff_arg, dtype=np.float64)
            if arr.ndim == 0:
                return float(arr)
            if arr.shape[0] == 0:
                return float(fallback)
            return float(np.mean(arr))

        eff0 = _mean_eff(det_eff_d0, self.det_eff_d0)
        eff1 = _mean_eff(det_eff_d1, self.det_eff_d1)

        if n_channels > 1 and modulation_index > EPS and mu_signal > EPS:
            mu_exp_signal_0 = max(0.0, eff0 * max(0.0, channel_transmittance) * mu_signal)
            mu_exp_signal_1 = max(0.0, eff1 * max(0.0, channel_transmittance) * mu_signal)

            n_cso = max(0, n_channels - 1) * 2
            if n_cso > 0:
                # QCNR formula from CATV CSO/CTB analysis (Phillips & Darcie 1997).
                qcnr = 16.0 / (modulation_index ** 2 * n_cso)
                mu_imd_base_0 = mu_exp_signal_0 / qcnr
                mu_imd_base_1 = mu_exp_signal_1 / qcnr

        # Optional WDM FWM contribution (uniform across detectors).
        if kwargs.get("include_wdm_fwm_noise", False):
            try:
                wdm_noise = float(calculate_wdm_fwm_noise(**kwargs))
                if math.isfinite(wdm_noise) and wdm_noise > 0.0:
                    mu_imd_base_0 += wdm_noise
                    mu_imd_base_1 += wdm_noise
            except Exception:
                if self.strict_mode:
                    raise
                logger.warning(
                    "WDM FWM noise calculation failed; ignoring contribution.",
                    exc_info=True,
                )

        return float(mu_imd_base_0), float(mu_imd_base_1)

    # ---- Public simulation API ----

    def simulate_detection(
        self,
        channel_transmittance: float,
        photon_numbers: NDArray[Any],
        rng: np.random.Generator,
        ideal_outcomes_d0: NDArray[Any],
        pulse_period_ns: float,
        return_diagnostics: bool = True,
        *,
        serialize_state: bool = True,
        return_photon_resolved: bool = False,
        **kwargs,
    ) -> Union[DetectionResult, "PhotonResolvedResult"]:
        """Run a detection simulation.

        Parameters
        ----------
        channel_transmittance:
            Channel power transmittance in ``[0, 1]``.
        photon_numbers:
            Integer-valued array of incoming photon counts per pulse.
        rng:
            ``numpy.random.Generator`` instance.
        ideal_outcomes_d0:
            Boolean array; ``True`` if the ideal outcome routes the
            signal to D0, ``False`` for D1.
        pulse_period_ns:
            Pulse period in nanoseconds; must be strictly positive.
        return_diagnostics:
            If ``True``, populate ``DetectionResult.diagnostics``.
        serialize_state:
            If ``True`` (default), populate
            ``DetectionResult.state_snapshot`` with the serialized
            runtime state.  If ``False``, ``state_snapshot`` is ``None``
            (review F25).
        return_photon_resolved:
            If ``True`` (review-v4 F-42), return a
            :class:`PhotonResolvedResult` instead of a
            :class:`DetectionResult`.  The photon-resolved result
            includes per-pulse ``arrivals``, ``target_d0``, and
            ``ideal_target_d0`` arrays so callers can compute decoy-
            state yields ``Y_n = P(click | n photons)`` directly.
            Only supported for the threshold path; raises for PNRD.

        Keyword arguments are forwarded to the selected path
        (threshold or PNRD) for visibility/dispersion/IMD parameters.
        The following kwargs are also accepted:

        - ``rng_buffer_mult`` (default :attr:`DetectorConstants.DEFAULT_RNG_BUFFER_MULT`)
        - ``out_buffer_mult`` (default :attr:`DetectorConstants.DEFAULT_OUT_BUFFER_MULT`)
        - ``co_buffer_mult``  (default :attr:`DetectorConstants.DEFAULT_CO_BUFFER_MULT`)
        - ``signal_split_ratio`` (default 1.0; review-v4 F-41/F-64).
          Probability that an arriving photon is routed to D0 (with
          the complement going to D1).  The default 1.0 corresponds to
          deterministic BB84 routing (polarizing beam splitter).  A
          value of 0.5 corresponds to a 50/50 beam splitter (BSM-like).
        - ``basis_selector`` (optional; review-v4 F-43).  Per-pulse
          array of 0 (Z basis) or 1 (X basis).  When supplied AND the
          detector has basis-specific efficiency fields set
          (``det_eff_d0_z``, ``det_eff_d0_x``, etc.), the per-pulse
          efficiency depends on the basis.  When not supplied, the
          symmetric ``det_eff_d0``/``det_eff_d1`` values are used.
        """
        # Review-v18 F-03: invalidate per-call dark-rate cache at the
        # start of each simulate_detection call so that
        # _calculate_dynamic_dark_rates is computed at most once per call
        # (the detector parameters are immutable within a call).
        self._cached_dynamic_dark_rates = None

        # --- Universal input validation -----------------------------------
        self._validate_simulation_inputs(
            channel_transmittance=channel_transmittance,
            photon_numbers=photon_numbers,
            rng=rng,
            ideal_outcomes_d0=ideal_outcomes_d0,
            pulse_period_ns=pulse_period_ns,
            return_diagnostics=return_diagnostics,
            serialize_state=serialize_state,
            return_photon_resolved=return_photon_resolved,
            **kwargs,
        )

        if self.detector_type == DetectorType.PNRD:
            if return_photon_resolved:
                raise ParameterValidationError(
                    "return_photon_resolved is not yet supported for the "
                    "PNRD path (review-v4 F-42).  Use the threshold path "
                    "(SPD/SNSPD) for photon-number-resolved statistics."
                )

            self._check_pnrd_state_compatibility()
            return self._simulate_pnrd(
                channel_transmittance=channel_transmittance,
                photon_numbers=photon_numbers,
                rng=rng,
                ideal_outcomes_d0=ideal_outcomes_d0,
                pulse_period_ns=pulse_period_ns,
                return_diagnostics=return_diagnostics,
                serialize_state=serialize_state,
                **kwargs,
            )

        self._check_threshold_state_compatibility()
        return self._simulate_threshold_with_retries(
            channel_transmittance=channel_transmittance,
            photon_numbers=photon_numbers,
            rng=rng,
            ideal_outcomes_d0=ideal_outcomes_d0,
            pulse_period_ns=pulse_period_ns,
            return_diagnostics=return_diagnostics,
            serialize_state=serialize_state,
            return_photon_resolved=return_photon_resolved,
            **kwargs,
        )

    def _check_pnrd_state_compatibility(self) -> None:
        """Raise or warn when switching threshold -> PNRD
        with non-trivial continuous-time state.

        The PNRD path is per-pulse and stateless w.r.t. dead time,
        afterpulsing, dark-count scheduling, and carry-over.  Switching
        from threshold to PNRD when those fields are non-trivial silently
        drops them.  We detect this and either raise (strict mode) or
        warn (non-strict mode) so the caller knows the transition is
        lossy.
        """
        state = self._internal_state
        # "Non-trivial" means ANY of the continuous-time fields has
        # been set by a previous threshold-path call.  The fresh-state
        # defaults are:
        #   - last_abs_fire_time_ns_d{0,1} == -np.inf
        #   - pending_ap_time_d{0,1} < 0
        #   - next_dc_time_d{0,1} < 0
        #   - carry_over_events == []
        #   - ap_depth_d{0,1} == 0
        non_trivial = (
            math.isfinite(state.last_abs_fire_time_ns_d0)
            or math.isfinite(state.last_abs_fire_time_ns_d1)
            or state.pending_ap_time_d0 >= 0.0
            or state.pending_ap_time_d1 >= 0.0
            or state.next_dc_time_d0 >= 0.0
            or state.next_dc_time_d1 >= 0.0
            or len(state.carry_over_events) > 0
            or state.ap_depth_d0 > 0
            or state.ap_depth_d1 > 0
        )
        if not non_trivial:
            return
        msg = (
            "Switching detector_type to PNRD with non-trivial "
            "continuous-time state from a previous threshold-path call. "
            "The PNRD path is per-pulse and stateless w.r.t. dead time, "
            "afterpulsing, dark-count scheduling, and carry-over; the "
            "following state will be SILENTLY DROPPED: "
            f"last_fire_d0={state.last_abs_fire_time_ns_d0!r}, "
            f"last_fire_d1={state.last_abs_fire_time_ns_d1!r}, "
            f"pending_ap_d0={state.pending_ap_time_d0!r}, "
            f"pending_ap_d1={state.pending_ap_time_d1!r}, "
            f"next_dc_d0={state.next_dc_time_d0!r}, "
            f"next_dc_d1={state.next_dc_time_d1!r}, "
            f"carry_over_count={len(state.carry_over_events)}, "
            f"ap_depth_d0={state.ap_depth_d0}, "
            f"ap_depth_d1={state.ap_depth_d1}.  Call reset_state() "
            "before switching types, or pick one detector type for the "
            "entire experiment (review-v6 C2/F2)."
        )
        if self.strict_mode:
            raise ParameterValidationError(msg)
        logger.warning(msg)

    def _check_threshold_state_compatibility(self) -> None:
        """Raise or warn when switching PNRD -> threshold
        with non-trivial absolute-time state.

        The PNRD path updates only ``total_time_processed_ns`` and
        ``afterpulse_total_count``.  Switching from PNRD to threshold
        sees ``last_fire = -inf``, ``next_dc = -1``, ``pending_ap = -1``
        (fresh-state defaults).  The dead-time / afterpulse / dark-count
        continuity is broken; the only state preserved is the absolute
        time clock.  This is documented behavior -- the PNRD path has
        no continuous-time concept -- but callers should be aware.
        """
        state = self._internal_state
        # Review-v8 R9-21: prefer the explicit ``last_path`` field
        # over the heuristic.
        if state.last_path == "threshold":
            # Previous call was threshold; continuous-time state is
            # valid.  No warning needed.
            return
        if state.last_path == "pnrd":
            # Previous call was PNRD.  The PNRD path updated
            # ``total_time_processed_ns`` but did NOT update
            # ``last_fire`` / ``pending_ap`` / ``next_dc`` /
            # ``carry_over`` -- those fields retain their fresh-state
            # defaults (-inf / -1).  Dead-time, afterpulse, and
            # dark-count continuity are LOST on the type switch.
            msg = (
                "Switching detector_type to threshold after a PNRD "
                "call (``last_path == 'pnrd'``).  The PNRD path "
                "updated total_time_processed_ns but did NOT update "
                "last_fire / pending_ap / next_dc / carry_over -- "
                "those fields retain their fresh-state defaults "
                "(-inf / -1).  Dead-time, afterpulse, and dark-count "
                "continuity are LOST on the type switch.  Call "
                "reset_state() before switching, or pick one "
                "detector type for the entire experiment (review-v6 "
                "C2/F2, review-v8 R9-21)."
            )
            if self.strict_mode:
                raise ParameterValidationError(msg)
            logger.warning(msg)
            return
        # ``last_path is None`` -- either no call has been made yet
        # (fresh state) or the state was loaded from a legacy v18-or-
        # earlier state file that did not serialize ``last_path``.
        # Fall back to the heuristic for backward compatibility.
        if state.total_time_processed_ns <= 0.0:
            return
        # If any continuous-time field is set, the previous call was
        # threshold (or the caller manually set the state), so no
        # warning is needed.
        has_continuous_state = (
            math.isfinite(state.last_abs_fire_time_ns_d0)
            or math.isfinite(state.last_abs_fire_time_ns_d1)
            or state.pending_ap_time_d0 >= 0.0
            or state.pending_ap_time_d1 >= 0.0
            or state.next_dc_time_d0 >= 0.0
            or state.next_dc_time_d1 >= 0.0
            or len(state.carry_over_events) > 0
        )
        if has_continuous_state:
            return
        msg = (
            "Switching detector_type to threshold after a PNRD call "
            "(heuristic detection: ``last_path`` is None but "
            "``total_time_processed_ns > 0`` and no continuous-time "
            "fields are set -- likely a legacy v18-or-earlier state "
            "file).  The PNRD path updated total_time_processed_ns "
            "but did NOT update last_fire / pending_ap / next_dc / "
            "carry_over -- those fields retain their fresh-state "
            "defaults (-inf / -1).  Dead-time, afterpulse, and "
            "dark-count continuity are LOST on the type switch.  "
            "Call reset_state() before switching, or pick one "
            "detector type for the entire experiment (review-v6 "
            "C2/F2, review-v8 R9-21 heuristic fallback)."
        )
        if self.strict_mode:
            raise ParameterValidationError(msg)
        logger.warning(msg)

    _PNRD_DC_STRICT_MODE_MAX_MU = 0.01

    def _pnrd_dc_approximation_valid(
        self, pulse_period_ns: float
    ) -> Tuple[bool, float, float]:
        """Report the per-pulse dark-count mean and the
        relative divergence between the PNRD per-pulse Poisson model and
        the threshold continuous-time Poisson model.

        Parameters
        ----------
        pulse_period_ns : float
            Pulse period in nanoseconds (per-call parameter).

        Returns
        -------
        Tuple[bool, float, float]
            ``(valid, mu_dc, rel_error)`` where:

            - ``valid`` is ``True`` iff ``mu_dc < 0.01`` (the per-pulse
              Poisson approximation is accurate to < 1% relative error).
            - ``mu_dc`` is the per-pulse dark-count mean
              ``rate * period * 1e-9`` (max of D0/D1 rates).
            - ``rel_error`` is a heuristic relative error estimate
              (``0`` for ``mu_dc = 0``, growing roughly as ``mu_dc**2``
              for small ``mu_dc`` and saturating at 1 for large ``mu_dc``).

        Notes
        -----
        The continuous-time Poisson model (threshold path) gives the
        SAME per-pulse count distribution as the per-pulse Poisson
        (PNRD) for non-gated operation, BUT it additionally resolves
        the *timing* of dark counts within the pulse period.  The PNRD
        path collapses this timing information into a single integer
        count.  For ``mu_dc << 1`` the per-pulse count is 0 or 1 (so
        the timing is irrelevant -- there is at most one event per
        pulse); for ``mu_dc >= 1`` the per-pulse count is non-sparse
        and the timing matters for dead-time / afterpulse
        interactions, which the PNRD path does not model.

        For gated detectors, the threshold path correctly scales dark
        counts by the duty factor ``gate_width / period``; the PNRD
        path does not (it would over-count).  This is why gated PNRD
        is always rejected.
        """
        pp = float(pulse_period_ns)
        if not math.isfinite(pp) or pp <= 0.0:
            return (False, float("inf"), 1.0)
        dr0, dr1 = self._calculate_dynamic_dark_rates()
        rate = max(dr0, dr1)
        mu_dc = max(0.0, rate) * pp * 1e-9
        # Heuristic relative-error estimate: for small mu_dc, the
        # probability of >= 2 dark counts in a pulse is ~mu_dc^2/2
        # (Poisson tail); this is the fraction of pulses where the
        # PNRD path loses information vs. the continuous-time path.
        # We cap at 1.0 for large mu_dc.
        rel_error = 1.0 - math.exp(-mu_dc) * (1.0 + mu_dc)
        rel_error = max(0.0, min(1.0, rel_error))
        valid = mu_dc < self._PNRD_DC_STRICT_MODE_MAX_MU
        return (valid, mu_dc, rel_error)

    def _check_pnrd_dc_approximation(self, pulse_period_ns: float) -> None:
        """Raise or warn when the PNRD per-pulse
        Poisson dark-count approximation diverges from the threshold
        continuous-time model.

        Called from ``_simulate_pnrd``.  In strict mode (the default),
        raises :class:`ParameterValidationError` when ``mu_dc >= 0.01``.
        """
        valid, mu_dc, rel_err = self._pnrd_dc_approximation_valid(pulse_period_ns)
        if valid:
            return
        msg = (
            f"PNRD per-pulse dark-count mean mu_dc={mu_dc:.4g} exceeds the "
            f"strict-mode threshold {self._PNRD_DC_STRICT_MODE_MAX_MU} "
            f"(relative model divergence ~= {rel_err:.4g}).  The PNRD path "
            f"uses a per-pulse Poisson model that loses timing "
            f"information for non-sparse dark counts; the threshold path "
            f"(SPD/SNSPD) uses a continuous-time Poisson model that "
            f"correctly resolves multiple dark counts per pulse.  "
            f"Switching detector types mid-experiment at this mu_dc would "
            f"silently bias QBER and gain estimates (review-v7 R7-03).  "
            f"Options: (a) switch to the threshold path (SPD/SNSPD); "
            f"(b) reduce the dark rate or pulse period; (c) set "
            f"strict_mode=False to silence this error (NOT recommended "
            f"for publication-bound work)."
        )
        if self.strict_mode:
            raise ParameterValidationError(msg)
        logger.warning(msg)

    def _validate_simulation_inputs(
        self,
        *,
        channel_transmittance: float,
        photon_numbers: NDArray[Any],
        rng: Any,
        ideal_outcomes_d0: NDArray[Any],
        pulse_period_ns: float,
        return_diagnostics: bool,
        serialize_state: bool,
        return_photon_resolved: bool = False,
        **kwargs,
    ) -> None:

        _KNOWN_KWARGS = {
            "rng_buffer_mult", "out_buffer_mult", "co_buffer_mult",
            "allow_nondefault_buffer",
            "signal_split_ratio", "per_photon_routing",
            "basis_selector",
            "distance_km", "linewidth_nm", "dispersion_parameter_ps_nm_km",
            "visibility_mismatch_dm", "visibility_bias_drift_psi1",
            "visibility_bias_drift_psi2",
            "N_channels", "modulation_index", "mu_signal",
            "include_wdm_fwm_noise",
            "flip_prob_override",
            "allow_xor_flip_formula",
            "ap_tail_tolerance",
        }
        unknown_kwargs = sorted(
            k for k in kwargs
            if k not in _KNOWN_KWARGS and not k.startswith("wdm_")
        )
        if unknown_kwargs:
            msg = (
                f"Unknown simulation kwarg(s): {unknown_kwargs}.  "
                f"Known kwargs: {sorted(_KNOWN_KWARGS)} (plus any "
                f"'wdm_'-prefixed kwarg forwarded to "
                f"calculate_wdm_fwm_noise).  Typos in kwarg names "
                f"silently produce wrong results (review-v5 F-50)."
            )
            if self.strict_mode:
                raise ParameterValidationError(msg)
            logger.warning(msg)

        if not isinstance(return_diagnostics, bool):
            raise ParameterValidationError(
                "Parameter 'return_diagnostics' must be a bool."
            )
        if not isinstance(serialize_state, bool):
            raise ParameterValidationError(
                "Parameter 'serialize_state' must be a bool."
            )
        if not isinstance(return_photon_resolved, bool):
            raise ParameterValidationError(
                "Parameter 'return_photon_resolved' must be a bool."
            )
        if not isinstance(rng, np.random.Generator):
            raise ParameterValidationError(
                "Parameter 'rng' must be a numpy.random.Generator."
            )
        ct = float(channel_transmittance)
        if not math.isfinite(ct) or ct < 0.0 or ct > 1.0:
            raise ParameterValidationError(
                f"Parameter 'channel_transmittance' must be in [0, 1]; got {ct!r}.",
                param_name="channel_transmittance",
                param_value=ct,
            )
        pp = float(pulse_period_ns)
        if not math.isfinite(pp) or pp <= 0.0:
            raise ParameterValidationError(
                f"Parameter 'pulse_period_ns' must be finite and strictly positive; got {pp!r}. "
                f"This typically means build_optical_source_config() did not pass "
                f"pulse_period_ns to OpticalSourceConfig, so the default (0.0) was used. "
                f"Ensure that build_optical_source_config() explicitly passes "
                f"pulse_period_ns=<value> to the OpticalSourceConfig constructor.",
                param_name="pulse_period_ns",
                param_value=pp,
            )

        # Gated mode: gate_width must be <= pulse_period.
        if self.gated:
            if self.gate_width_ns > pp:
                raise ParameterValidationError(
                    f"gate_width_ns ({self.gate_width_ns}) must not exceed "
                    f"pulse_period_ns ({pp})."
                )
            # Review F11 fix: gate_offset must be < pulse_period.
            if self.gate_offset_ns >= pp:
                raise ParameterValidationError(
                    f"gate_offset_ns ({self.gate_offset_ns}) must be strictly "
                    f"less than pulse_period_ns ({pp})."
                )
            # Gate window must fit inside the period.
            if self.gate_offset_ns + self.gate_width_ns > pp:
                raise ParameterValidationError(
                    f"gate_offset_ns + gate_width_ns "
                    f"({self.gate_offset_ns} + {self.gate_width_ns} = "
                    f"{self.gate_offset_ns + self.gate_width_ns}) must not exceed "
                    f"pulse_period_ns ({pp})."
                )

        # distance_km / linewidth_nm / dispersion_parameter may be
        # supplied via kwargs for chromatic-dispersion broadening.
        for k in ("distance_km", "linewidth_nm", "dispersion_parameter_ps_nm_km"):
            if k in kwargs:
                v = float(kwargs[k])
                if not math.isfinite(v) or v < 0.0:
                    raise ParameterValidationError(
                        f"Parameter '{k}' must be finite and non-negative; got {v!r}."
                    )

        if self.detector_type == DetectorType.PNRD:
            dispersion_kwargs_present = any(
                k in kwargs
                for k in ("distance_km", "linewidth_nm", "dispersion_parameter_ps_nm_km")
            )
            # Treat distance_km == 0 / linewidth_nm == 0 as "no
            # dispersion" -- do not raise.
            dispersion_active = dispersion_kwargs_present and (
                float(kwargs.get("distance_km", 0.0)) > 0.0
                and float(kwargs.get("linewidth_nm", 0.0)) > 0.0
                and float(kwargs.get("dispersion_parameter_ps_nm_km", 0.0)) > 0.0
            )
            if dispersion_active:
                if self.strict_mode:
                    raise ParameterValidationError(
                        "Strict mode: PNRD path does not currently model "
                        "chromatic-dispersion broadening.  Either switch "
                        "to a threshold (SPD/SNSPD) detector or set "
                        "strict_mode=False to silently ignore the "
                        "dispersion kwargs (review-v4 F-06)."
                    )
                logger.warning(
                    "PNRD path: dispersion kwargs supplied but ignored "
                    "(review-v4 F-06).  Set strict_mode=True to raise."
                )

            ssr = float(kwargs.get("signal_split_ratio", 1.0))
            if not is_close(ssr, 1.0) and ssr >= 0.0 and ssr <= 1.0:
                if self.strict_mode:
                    raise ParameterValidationError(
                        f"Strict mode: PNRD path does not implement BSM-"
                        f"like symmetric routing (signal_split_ratio = "
                        f"{ssr!r} != 1.0).  The PNRD path uses "
                        f"deterministic routing (target_d0 -> D0, else "
                        f"D1); a non-trivial signal_split_ratio would "
                        f"be silently ignored.  Either switch to a "
                        f"threshold detector (which supports "
                        f"signal_split_ratio < 1) or set "
                        f"strict_mode=False to silence this error "
                        f"(review-v6 F19)."
                    )
                logger.warning(
                    "PNRD path: signal_split_ratio=%g supplied but "
                    "ignored (deterministic routing only).  Set "
                    "strict_mode=True to raise (review-v6 F19).",
                    ssr,
                )

        # Buffer-sizing kwargs (testing / advanced use).
        for k in ("rng_buffer_mult", "out_buffer_mult", "co_buffer_mult"):
            if k in kwargs:
                v = float(kwargs[k])
                if not math.isfinite(v) or v <= 0.0:
                    raise ParameterValidationError(
                        f"Parameter '{k}' must be finite and strictly positive; got {v!r}."
                    )

        allow_nondefault_buffer = bool(kwargs.get("allow_nondefault_buffer", False))
        if "rng_buffer_mult" in kwargs and self.strict_mode and not allow_nondefault_buffer:
            supplied = float(kwargs["rng_buffer_mult"])
            if not is_close(supplied, _CONST.DEFAULT_RNG_BUFFER_MULT):
                raise ParameterValidationError(
                    f"rng_buffer_mult ({supplied}) differs from the default "
                    f"({_CONST.DEFAULT_RNG_BUFFER_MULT}).  rng_buffer_mult is "
                    f"a BEHAVIORAL parameter (review-v7 R7-01): the kernel "
                    f"consumes floats from a pre-drawn array whose length "
                    f"depends on rng_buffer_mult, so different buffer sizes "
                    f"produce different random sequences for the same seed.  "
                    f"For publication-bound work, fix rng_buffer_mult at the "
                    f"default value and document it as a fixed parameter.  "
                    f"To bypass this guard for testing, pass "
                    f"allow_nondefault_buffer=True."
                )

        pn = np.asarray(photon_numbers, copy=False)
        if pn.ndim != 1:
            raise ParameterValidationError(
                "Parameter 'photon_numbers' must be a 1D array."
            )
        if np.issubdtype(pn.dtype, np.floating):
            if not np.all(np.isfinite(pn)) or np.any(pn < 0.0):
                raise ParameterValidationError(
                    "Parameter 'photon_numbers' must be non-negative finite; "
                    "negative or NaN/Inf values are not allowed."
                )
            if not np.all(np.equal(np.mod(pn, 1.0), 0.0)):
                raise ParameterValidationError(
                    "Parameter 'photon_numbers' must be integer-valued."
                )
        elif np.issubdtype(pn.dtype, np.integer):
            if np.any(pn < 0):
                raise ParameterValidationError(
                    "Parameter 'photon_numbers' must be non-negative."
                )
        else:
            raise ParameterValidationError(
                f"Parameter 'photon_numbers' has unsupported dtype {pn.dtype!r}; "
                "expected integer or float."
            )

        oo = np.asarray(ideal_outcomes_d0, copy=False)
        if oo.ndim != 1:
            raise ParameterValidationError(
                "Parameter 'ideal_outcomes_d0' must be a 1D array."
            )
        if oo.shape[0] != pn.shape[0]:
            raise ParameterValidationError(
                "Parameter 'ideal_outcomes_d0' must have the same length as "
                "'photon_numbers'."
            )

        if "basis_selector" in kwargs:
            bs = np.asarray(kwargs["basis_selector"], copy=False)
            if bs.ndim != 1:
                raise ParameterValidationError(
                    "Parameter 'basis_selector' must be a 1D array."
                )
            if bs.shape[0] != pn.shape[0]:
                raise ParameterValidationError(
                    "Parameter 'basis_selector' must have the same length as "
                    "'photon_numbers'."
                )
            # Each entry must be 0 (Z basis) or 1 (X basis).
            if not np.all(np.isin(bs, (0, 1))):
                raise ParameterValidationError(
                    "Parameter 'basis_selector' must contain only 0 (Z) or 1 (X)."
                )
        else:
            has_basis_eff = (
                getattr(self, "det_eff_d0_z", None) is not None
                or getattr(self, "det_eff_d0_x", None) is not None
                or getattr(self, "det_eff_d1_z", None) is not None
                or getattr(self, "det_eff_d1_x", None) is not None
            )
            if has_basis_eff:
                msg = (
                    "Basis-dependent detector efficiency fields "
                    "(det_eff_d{0,1}_{z,x}) are configured on the "
                    "detector, but no ``basis_selector`` kwarg was "
                    "supplied to ``simulate_detection``.  The "
                    "simulation will fall back to the scalar "
                    "``det_eff_d{0,1}`` values, silently dropping "
                    "the basis-dependent efficiency.  Pass a "
                    "``basis_selector`` array (per-pulse 0 for Z, 1 "
                    "for X) to use the basis-dependent values, or set "
                    "the basis-specific fields to None if scalar "
                    "efficiency is intended."
                )
                if self.strict_mode:
                    raise ParameterValidationError(msg)
                logger.warning(msg)

        if "flip_prob_override" in kwargs:
            fpo = kwargs["flip_prob_override"]
            if fpo is None:
                pass  # treat None as "not supplied"
            else:
                try:
                    fpo_f = float(fpo)
                except (TypeError, ValueError):
                    raise ParameterValidationError(
                        f"Parameter 'flip_prob_override' must be a float in "
                        f"[0, 1] or None; got {fpo!r}."
                    )
                if not math.isfinite(fpo_f) or fpo_f < 0.0 or fpo_f > 1.0:
                    raise ParameterValidationError(
                        f"Parameter 'flip_prob_override' must be a finite "
                        f"probability in [0, 1]; got {fpo_f!r}."
                    )

        if (
            self.strict_mode
            and "flip_prob_override" not in kwargs
            and not bool(kwargs.get("allow_xor_flip_formula", False))
            and (self.misalignment > 0.0 or self.qber_intrinsic > 0.0)
        ):
            raise ParameterValidationError(
                f"Strict mode: the XOR-combined flip probability "
                f"``m*(1-q) + (1-m)*q`` is being used without a direct "
                f"``flip_prob_override`` (review-v8 R9-03).  The XOR "
                f"formula assumes statistical independence between "
                f"misalignment (m={self.misalignment}) and "
                f"QBER/visibility errors (qber_intrinsic="
                f"{self.qber_intrinsic}), which is physically wrong "
                f"for BB84 (the two are correlated through visibility "
                f"degradation).  The XOR formula can overestimate the "
                f"combined error rate by up to ~2x in the small-error "
                f"limit, silently biasing QBER and decoy-state yield "
                f"estimates.  Options: (a) supply "
                f"``flip_prob_override=<float in [0,1]>`` to use a "
                f"directly-computed flip probability (RECOMMENDED for "
                f"publication-bound work); (b) supply "
                f"``allow_xor_flip_formula=True`` to explicitly opt in "
                f"to the XOR formula (suitable for exploratory work or "
                f"matching legacy-v17 results); (c) set "
                f"``strict_mode=False`` on the detector (NOT "
                f"recommended for publication-bound work)."
            )

        if (
            getattr(self, "afterpulse_prob", 0.0) > 0.0
            and getattr(self, "afterpulse_lifetime_ns", 0.0) <= 0.0
        ):
            raise ParameterValidationError(
                f"afterpulse_prob ({self.afterpulse_prob}) > 0 requires "
                f"afterpulse_lifetime_ns > 0 (got "
                f"{self.afterpulse_lifetime_ns}).  A zero or negative "
                f"AP lifetime causes division-by-zero in the "
                f"geometric/exponential AP delay sampler."
            )

        if "ap_tail_tolerance" in kwargs:
            ap_tail_tolerance = float(kwargs["ap_tail_tolerance"])
            if (
                not math.isfinite(ap_tail_tolerance)
                or ap_tail_tolerance <= 0.0
                or ap_tail_tolerance >= 1.0
            ):
                raise ParameterValidationError(
                    f"ap_tail_tolerance must be a finite probability in "
                    f"(0, 1); got {ap_tail_tolerance!r} (F-20)."
                )

        if self.max_count_rate_hz is not None:
            dr0, dr1 = self._calculate_dynamic_dark_rates()
            dark_only_rate = dr0 + dr1
            if dark_only_rate > self.max_count_rate_hz:
                msg = (
                    f"Dark-count rate alone ({dark_only_rate:.4g} Hz) "
                    f"exceeds max_count_rate_hz "
                    f"({self.max_count_rate_hz:.4g} Hz) (F-14).  The "
                    f"detector's electronic bandwidth cannot sustain "
                    f"even the dark counts; the simulation result "
                    f"would be non-physical.  Reduce the dark rate, "
                    f"raise max_count_rate_hz, or set strict_mode=False "
                    f"to silence this error."
                )
                if self.strict_mode:
                    raise ParameterValidationError(msg)
                logger.warning(msg)

    def _simulate_threshold_with_retries(
        self,
        *,
        channel_transmittance: float,
        photon_numbers: NDArray[Any],
        rng: np.random.Generator,
        ideal_outcomes_d0: NDArray[Any],
        pulse_period_ns: float,
        return_diagnostics: bool,
        serialize_state: bool,
        return_photon_resolved: bool = False,
        **kwargs,
    ) -> DetectionResult:
        """Threshold-path driver with adaptive buffer-growth retries.
        Each retry multiplies only the buffer class that exhausted;
        attempts are bounded and surfaced via ``diag.num_retries``.
        """
        max_retries = _CONST.MAX_RETRIES
        rng_multiplier = float(kwargs.pop("rng_buffer_mult", _CONST.DEFAULT_RNG_BUFFER_MULT))
        out_multiplier = float(kwargs.pop("out_buffer_mult", _CONST.DEFAULT_OUT_BUFFER_MULT))
        co_multiplier = float(kwargs.pop("co_buffer_mult", _CONST.DEFAULT_CO_BUFFER_MULT))
        # Review-v7 R7-01: ``allow_nondefault_buffer`` is a meta-kwarg
        # consumed by ``_validate_simulation_inputs`` (it bypasses the
        # strict-mode guard on non-default ``rng_buffer_mult``).  Pop
        # it here so it does not leak into ``_simulate_event_driven`` /
        # ``_calculate_imd_intensity`` via ``**kwargs``.
        kwargs.pop("allow_nondefault_buffer", None)
        attempts = 0
        # Review-v15 F-02: track the initial RNG multiplier so we can
        # detect when a retry would change the buffer size (and thus
        # the RNG sequence).  In strict mode, retries that change the
        # buffer size are rejected because they break reproducibility.
        _initial_rng_multiplier = rng_multiplier

        snap = _copy_rng_state(rng.bit_generator.state)
        last_err: Optional[RuntimeError] = None
        for attempt in range(max_retries):
            attempts = attempt + 1
            if attempt > 0:
                # Re-apply the pre-call snapshot before each retry so
                # the kernel sees the same RNG state regardless of how
                # many retries occurred (review C3 in earlier reviews).
                rng.bit_generator.state = _copy_rng_state(snap)
                # Review-v15 F-02: if the RNG buffer size changed due
                # to a retry, the kernel will consume a different
                # random sequence from the pre-drawn buffer, breaking
                # reproducibility.  In strict mode, raise instead of
                # silently proceeding with a different sequence.
                if (
                    self.strict_mode
                    and rng_multiplier != _initial_rng_multiplier
                ):
                    raise ParameterValidationError(
                        f"RNG buffer exhausted on first attempt; retry "
                        f"with rng_buffer_mult={rng_multiplier:.1f} "
                        f"(was {_initial_rng_multiplier:.1f}) would "
                        f"draw a different number of random floats, "
                        f"changing the kernel's random sequence and "
                        f"breaking reproducibility (F-02).  Increase "
                        f"rng_buffer_mult to {_initial_rng_multiplier * _CONST.RETRY_RNG_GROWTH:.1f} "
                        f"or higher to avoid the retry, or set "
                        f"strict_mode=False to allow the retry (NOT "
                        f"recommended for publication-bound work)."
                    )
            try:
                result = self._simulate_event_driven(
                    channel_transmittance=channel_transmittance,
                    photon_numbers=photon_numbers,
                    rng=rng,
                    ideal_outcomes_d0=ideal_outcomes_d0,
                    pulse_period_ns=pulse_period_ns,
                    return_diagnostics=return_diagnostics,
                    rng_buffer_mult=rng_multiplier,
                    out_buffer_mult=out_multiplier,
                    co_buffer_mult=co_multiplier,
                    serialize_state=serialize_state,
                    return_photon_resolved=return_photon_resolved,
                    **kwargs,
                )
                if return_diagnostics and result.diagnostics is not None:
                    result = result._replace(
                        diagnostics=replace(
                            result.diagnostics,
                            num_retries=attempts - 1,
                            buffer_resized=(attempts > 1),
                        )
                    )
                if not serialize_state:
                    result = result._replace(state_snapshot=None)
                return result
            except RuntimeError as e:
                last_err = e
                err_msg = str(e)
                if "RNG buffer exhausted" in err_msg and attempt < max_retries - 1:
                    logger.warning("RNG buffer exhausted; retrying with larger RNG buffer (attempt %d/%d).",
                                   attempt + 1, max_retries)
                    rng_multiplier *= _CONST.RETRY_RNG_GROWTH
                    continue
                if "Output buffer exhausted" in err_msg and attempt < max_retries - 1:
                    logger.warning("Output buffer exhausted; retrying with larger output buffer (attempt %d/%d).",
                                   attempt + 1, max_retries)
                    out_multiplier *= _CONST.RETRY_OUT_GROWTH
                    continue
                if "Carry-over buffer exhausted" in err_msg and attempt < max_retries - 1:
                    logger.warning("Carry-over buffer exhausted; retrying with larger carry-over buffer (attempt %d/%d).",
                                   attempt + 1, max_retries)
                    co_multiplier *= _CONST.RETRY_CO_GROWTH
                    continue
                # On any other RuntimeError, restore the snapshot so the
                # caller's rng is left in the pre-call state (matches the
                # old guard semantics for exception paths).
                rng.bit_generator.state = _copy_rng_state(snap)
                raise

        # All retries exhausted -- restore the pre-call snapshot so the
        # caller's rng is left in the pre-call state (matches the old
        # guard semantics for the failure path).
        rng.bit_generator.state = _copy_rng_state(snap)
        raise RuntimeError(
            "Simulation failed after max retries due to buffer exhaustion."
        )

    # ---- PNRD path ----

    def _simulate_pnrd(
        self,
        *,
        channel_transmittance: float,
        photon_numbers: NDArray[Any],
        rng: np.random.Generator,
        ideal_outcomes_d0: NDArray[Any],
        pulse_period_ns: float,
        return_diagnostics: bool,
        serialize_state: bool,
        **kwargs,
    ) -> DetectionResult:
        """Photon-number-resolving path.

        Physical model
        ---------------
        - ``photon_numbers`` are integer photon counts per pulse from
          the source.
        - Channel thinning: ``arrivals ~ Binomial(photon_numbers, eff_trans)``
          where ``eff_trans = channel_transmittance * intensity_factor``.
        - Routing: each arriving photon is routed to the ideal detector
          (D0 if ``ideal_outcomes_d0`` is True else D1), with the
          routing flipped independently by ``misalignment`` and by the
          QBER/visibility error probability (XOR-combined).
        - Detection: ``counts_d0 ~ Binomial(arrivals_d0, det_eff_d0)``
          and analogously for D1.
        - Dark counts: per-pulse Poisson with mean ``rate * period``
          (consistent within PNRD, but differs from the continuous-time
          model in the threshold path -- see module docstring, "Mixed
          pulse-index and continuous-time domains", review P12).
        - IMD/FWM: per-pulse Poisson with detector-specific means.
        - Misalignment and QBER are applied **consistently with the
          threshold path** so switching ``detector_type`` does not
          silently remove the error model.
        """
        num_pulses = int(ideal_outcomes_d0.shape[0])

        kwargs.pop("allow_nondefault_buffer", None)
        # ``rng_buffer_mult`` / ``out_buffer_mult`` / ``co_buffer_mult``
        # are threshold-path-only; pop them silently for API symmetry.
        kwargs.pop("rng_buffer_mult", None)
        kwargs.pop("out_buffer_mult", None)
        kwargs.pop("co_buffer_mult", None)
        per_photon_routing_pnrd = bool(kwargs.pop("per_photon_routing", True))
        # Review-v17 C1/C2 fix: store popped values so the PNRD path
        # actually uses them, rather than discarding and then checking
        # kwargs (which always fails because the keys were already
        # removed).
        flip_prob_override_pnrd = kwargs.pop("flip_prob_override", None)
        allow_xor_pnrd = kwargs.pop("allow_xor_flip_formula", None)

        self._check_pnrd_dc_approximation(float(pulse_period_ns))

        if self.strict_mode:
            policy_is_no_op = (
                self.double_click_policy == DoubleClickPolicy.DISCARD
                or _is_rogers_policy(self.double_click_policy)
            )
            if not policy_is_no_op:
                raise ParameterValidationError(
                    f"Strict mode: double_click_policy="
                    f"{self.double_click_policy!r} is set on a PNRD "
                    f"detector, but the PNRD path does not implement "
                    f"double-click resolution (review-v8 R9-25).  PNRD "
                    f"returns integer counts per pulse; simultaneous "
                    f"D0&D1 counts are reported as-is.  Either switch "
                    f"to a threshold detector (SPD/SNSPD) for "
                    f"double-click resolution, set "
                    f"double_click_policy=DoubleClickPolicy.DISCARD "
                    f"(treated as 'no resolution' on PNRD), or set "
                    f"strict_mode=False to silence this error."
                )

        pn_check = np.asarray(photon_numbers, dtype=np.int64)
        has_multiphoton = bool(np.any(pn_check > 1))

        if (
            has_multiphoton
            and self.strict_mode
            and not per_photon_routing_pnrd
        ):
            raise ParameterValidationError(
                f"Strict mode: PNRD path received multi-photon input "
                f"(max photon_number = {int(np.max(pn_check))}) with "
                f"per_photon_routing=False, but per-PULSE routing "
                f"(all photons in a multi-photon pulse go to the same "
                f"detector) silently biases decoy-state yields Y_n for "
                f"n >= 2 (F-06 / review-v8 R9-01).  Options: "
                f"(a) pass per_photon_routing=True (the default, "
                f"physically correct); (b) switch to a threshold "
                f"detector (DetectorType.SPD or DetectorType.SNSPD); "
                f"(c) restrict input to single-photon pulses "
                f"(max photon_number <= 1); (d) set strict_mode=False "
                f"to acknowledge the per-pulse routing approximation "
                f"(NOT recommended for publication-bound decoy-state "
                f"work)."
            )

        # --- Channel / visibility preparation ---
        dm = float(kwargs.get("visibility_mismatch_dm", 0.0))
        dpsi1 = float(kwargs.get("visibility_bias_drift_psi1", 0.0))
        dpsi2 = float(kwargs.get("visibility_bias_drift_psi2", 0.0))

        intensity_factor = calculate_intensity_mismatch_factor(dm, dpsi1, dpsi2)
        eff_trans = clamp_probability(float(channel_transmittance) * float(intensity_factor))

        visibility = clamp_probability(float(calculate_visibility(dm, dpsi1, dpsi2)))
        p_qber = clamp_probability(self.qber_intrinsic + (1.0 - visibility) / 2.0)
        if p_qber > 0.5:
            if self.strict_mode:
                raise ParameterValidationError(
                    f"p_qber ({p_qber}) exceeds 0.5; channel is worse than "
                    f"random guessing.  Check qber_intrinsic "
                    f"({self.qber_intrinsic}) and visibility ({visibility})."
                )
            logger.warning(
                "p_qber (%g) exceeds 0.5; clamping to 0.5 (review-v4 F-14/F-56).",
                p_qber,
            )
            p_qber = 0.5

        # Review-v17 C1/C2 fix: use the stored local variable (popped
        # above) rather than re-checking kwargs after the key was removed.
        if flip_prob_override_pnrd is not None:
            p_combined_flip = clamp_probability(float(flip_prob_override_pnrd))
        else:
            p_combined_flip = clamp_probability(
                self.misalignment * (1.0 - p_qber) + (1.0 - self.misalignment) * p_qber
            )

        # --- Channel thinning ---
        photon_numbers_i = np.asarray(photon_numbers, dtype=np.int64)
        arrivals = rng.binomial(photon_numbers_i, eff_trans)

        # --- Routing of arrivals to D0 / D1 ---
        if per_photon_routing_pnrd and has_multiphoton:
            # Per-photon routing: draw ``n_total = sum(arrivals)`` flips.
            n_total = int(np.sum(arrivals))
            if n_total > 0:
                photon_flips = rng.random(n_total) < p_combined_flip
                # Build the per-photon ideal target by repeating.
                ideal_per_pulse = np.asarray(ideal_outcomes_d0, dtype=np.bool_)
                photon_ideal = np.repeat(ideal_per_pulse, arrivals)
                photon_actual_to_d0 = photon_ideal ^ photon_flips
                # Per-photon Binomial: each photon independently goes to
                # its target detector.  The per-pulse arrival count for
                # D0 is the sum of (target == D0) over the pulse's
                # photons -- but each photon has a fixed target (0 or 1),
                # so the count is just the sum of the per-photon target
                # mask grouped by pulse.  We use ``np.add.reduceat`` for
                # efficiency.
                pulse_idx = np.repeat(np.arange(num_pulses), arrivals)
                # arrivals_d0[b] = number of photons in pulse b whose
                # target is D0.
                arrivals_d0 = np.zeros(num_pulses, dtype=np.int64)
                np.add.at(arrivals_d0, pulse_idx, photon_actual_to_d0.astype(np.int64))
                arrivals_d1 = arrivals - arrivals_d0
                # Per-pulse flip aggregation for diagnostics.
                photon_flips_per_pulse = np.zeros(num_pulses, dtype=np.bool_)
                np.logical_or.at(photon_flips_per_pulse, pulse_idx, photon_flips)
                flips_mask = photon_flips_per_pulse
            else:
                # No arrivals; degenerate.
                flips_mask = np.zeros(num_pulses, dtype=np.bool_)
                arrivals_d0 = np.zeros(num_pulses, dtype=np.int64)
                arrivals_d1 = np.zeros(num_pulses, dtype=np.int64)
        else:
            # Per-pulse routing (legacy, or single-photon input where
            # per-pulse and per-photon routing are equivalent).
            target_d0 = np.asarray(ideal_outcomes_d0, dtype=np.bool_).copy()
            flips_mask = rng.random(num_pulses) < p_combined_flip
            target_d0[flips_mask] = ~target_d0[flips_mask]
            # Each photon independently goes to D0 with probability
            # ``route_p_d0`` (1.0 where target is D0, 0.0 where target
            # is D1).  Binomial draw produces the per-pulse arrival
            # count for D0; the rest go to D1.  This guarantees no
            # cross-detector double counting.
            route_p_d0 = target_d0.astype(np.float64)
            arrivals_d0 = rng.binomial(arrivals, route_p_d0)
            arrivals_d1 = arrivals - arrivals_d0

        # --- Detection (per-detector quantum efficiency) ---
        det_counts_d0 = rng.binomial(arrivals_d0, self.det_eff_d0)
        det_counts_d1 = rng.binomial(arrivals_d1, self.det_eff_d1)

        # --- IMD / FWM ---
        mu_noise_d0, mu_noise_d1 = self._calculate_imd_intensity(
            float(channel_transmittance), rng, num_pulses, **kwargs
        )
        imd_counts_d0 = rng.poisson(mu_noise_d0, size=num_pulses)
        imd_counts_d1 = rng.poisson(mu_noise_d1, size=num_pulses)

        # --- Dark counts (per-pulse Poisson, PNRD-specific) ---
        dr0, dr1 = self._calculate_dynamic_dark_rates()
        mu_dc_d0 = max(0.0, dr0) * float(pulse_period_ns) * 1e-9
        mu_dc_d1 = max(0.0, dr1) * float(pulse_period_ns) * 1e-9
        dark_counts_d0 = rng.poisson(mu_dc_d0, size=num_pulses)
        dark_counts_d1 = rng.poisson(mu_dc_d1, size=num_pulses)

        final_d0 = det_counts_d0 + dark_counts_d0 + imd_counts_d0
        final_d1 = det_counts_d1 + dark_counts_d1 + imd_counts_d1

        # Cast to int64 for a stable downstream contract.
        final_d0 = final_d0.astype(np.int64, copy=False)
        final_d1 = final_d1.astype(np.int64, copy=False)

        # --- State update ---

        self._internal_state.total_time_processed_ns += num_pulses * float(pulse_period_ns)

        # PNRD generates no afterpulses (``afterpulse_prob > 0`` is
        # rejected for PNRD at construction in strict mode), so the
        # ``afterpulse_total_count`` counter is intentionally left
        # unchanged by this path.  The threshold path advances it from
        # the kernel's ``DIAG_AFTERPULSE`` diagnostic.
        self._internal_state.last_path = "pnrd"

        diag = None
        if return_diagnostics:
            ideal_arr = np.asarray(ideal_outcomes_d0, dtype=np.bool_)
            combined_flips_with_arrivals = int(np.sum(flips_mask & (arrivals > 0)))
            # Pure-QBER flips: arrivals > 0, ideal target was D0, got
            # flipped to D1 by the COMBINED misalignment+QBER process.
            # This is a subset of ``combined_flips_with_arrivals``.
            qber_flips_only = int(np.sum(flips_mask & (arrivals > 0) & ideal_arr))
            diag = DetectionDiagnostics(
                combined_flips_with_arrivals=combined_flips_with_arrivals,
                qber_flips_only=qber_flips_only,
                # Convention: qber_flips counts arrivals whose ideal
                # target was D0 and got flipped to D1 (review F54).
                # Retained for backward compat; equals qber_flips_only.
                qber_flips=qber_flips_only,
                imd_click_events=int(np.sum(imd_counts_d0) + np.sum(imd_counts_d1)),
                dark_count_events=int(np.sum(dark_counts_d0) + np.sum(dark_counts_d1)),
            )

        return DetectionResult(
            final_d0,
            final_d1,
            diag,
            self.get_state() if serialize_state else None,
        )

    # ---- Threshold (event-driven) path ----

    def _probe_rng_buffer_sensitivity(self, **sim_kwargs) -> bool:
        """Test hook (review-v4 F-54; review-v8 R9-07/R9-14 rename):
        probe whether ``rng_buffer_mult`` affects simulation results.

        Runs the same simulation twice with ``rng_buffer_mult = 10.0``
        (default) and ``rng_buffer_mult = 20.0`` (larger buffer) and
        compares the click arrays.  Returns ``True`` if the click
        arrays are bit-identical across the two buffer sizes, ``False``
        otherwise.

        .. warning::

            The invariant probed by this hook is NOT currently
            enforced.  ``rng_buffer_mult`` is a BEHAVIORAL parameter
            (review-v7 R7-01, review-v8 R9-07): the kernel consumes
            floats from a pre-drawn ``rng_floats`` array whose length
            depends on ``rng_buffer_mult``; different buffer sizes
            therefore produce different random sequences for the
            kernel even when no buffer exhaustion occurs.

            For publication-bound work, callers MUST fix
            ``rng_buffer_mult`` at the default value (10.0) and
            document it as a fixed parameter of the simulation.  The
            strict-mode guard in ``_validate_simulation_inputs``
            (review-v7 R7-01) raises if a non-default
            ``rng_buffer_mult`` is supplied without
            ``allow_nondefault_buffer=True``; this hook passes that
            flag explicitly so it can probe the behavioral
            dependency.

        Parameters
        ----------
        **sim_kwargs
            Forwarded to :meth:`simulate_detection`.  Required keys:
            ``rng``, ``channel_transmittance``, ``photon_numbers``,
            ``ideal_outcomes_d0``, ``pulse_period_ns``.

        Returns
        -------
        bool
            ``True`` if click arrays are bit-identical across
            ``rng_buffer_mult = 10.0`` and ``rng_buffer_mult = 20.0``;
            ``False`` otherwise.  In the current implementation this
            typically returns ``False`` (the invariant is NOT
            enforced); callers should treat the return value as a
            MEASUREMENT of the sensitivity, not a validation.
        """
        required = ("rng", "channel_transmittance", "photon_numbers",
                    "ideal_outcomes_d0", "pulse_period_ns")
        for k in required:
            if k not in sim_kwargs:
                raise ValueError(
                    f"_probe_rng_buffer_sensitivity requires a {k!r} kwarg."
                )

        # Snapshot the original RNG state so both runs see the same
        # random sequence.
        original_rng = sim_kwargs["rng"]
        original_state = original_rng.bit_generator.state

        # Run 1: default buffer mult.
        original_rng.bit_generator.state = _copy_rng_state(original_state)
        kwargs1 = dict(sim_kwargs)
        kwargs1["rng_buffer_mult"] = _CONST.DEFAULT_RNG_BUFFER_MULT
        r1 = self.simulate_detection(**kwargs1)

        # Run 2: larger buffer mult.
        # Pass ``allow_nondefault_buffer=True`` to bypass the strict-
        # mode guard on non-default ``rng_buffer_mult``.  This test
        # hook EXISTS to document the buffer-size behavioral
        # dependency, so it must be allowed to set a non-default
        # value.
        original_rng.bit_generator.state = _copy_rng_state(original_state)
        kwargs2 = dict(sim_kwargs)
        kwargs2["rng_buffer_mult"] = 2.0 * _CONST.DEFAULT_RNG_BUFFER_MULT
        kwargs2["allow_nondefault_buffer"] = True
        r2 = self.simulate_detection(**kwargs2)

        # Restore the original RNG state (so the caller's rng is not
        # perturbed by the test hook).
        original_rng.bit_generator.state = _copy_rng_state(original_state)

        return bool(
            np.array_equal(r1.click0, r2.click0)
            and np.array_equal(r1.click1, r2.click1)
        )

    def _validate_buffer_independence(self, **sim_kwargs) -> bool:
        """Deprecated alias for :meth:`_probe_rng_buffer_sensitivity`.

        .. deprecated:: v19 (review-v8 R9-07/R9-14)
            The name ``_validate_buffer_independence`` was misleading:
            the invariant is NOT enforced (``rng_buffer_mult`` is a
            behavioral parameter).  Use
            :meth:`_probe_rng_buffer_sensitivity` instead, which
            accurately describes what the hook does (probe, not
            validate).
        """
        warnings.warn(
            "_validate_buffer_independence is deprecated "
            "(review-v8 R9-07/R9-14).  The name was misleading: the "
            "invariant is NOT enforced (rng_buffer_mult is a "
            "behavioral parameter).  Use _probe_rng_buffer_sensitivity "
            "instead, which accurately describes what the hook does.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._probe_rng_buffer_sensitivity(**sim_kwargs)

    def _simulate_event_driven(
        self,
        *,
        channel_transmittance: float,
        photon_numbers: NDArray[Any],
        rng: np.random.Generator,
        ideal_outcomes_d0: NDArray[Any],
        pulse_period_ns: float,
        return_diagnostics: bool,
        rng_buffer_mult: float = _CONST.DEFAULT_RNG_BUFFER_MULT,
        out_buffer_mult: float = _CONST.DEFAULT_OUT_BUFFER_MULT,
        co_buffer_mult: float = _CONST.DEFAULT_CO_BUFFER_MULT,
        serialize_state: bool = True,
        return_photon_resolved: bool = False,
        **kwargs,
    ) -> Union[DetectionResult, "PhotonResolvedResult"]:
        """Single-attempt event-driven simulation (no retry logic).

        Raises :class:`RuntimeError` on buffer exhaustion; the caller
        (:meth:`_simulate_threshold_with_retries`) handles retries with
        RNG state restoration.
        """
        state = self._internal_state
        num_pulses = int(ideal_outcomes_d0.shape[0])

        flip_prob_override = kwargs.pop("flip_prob_override", None)
        kwargs.pop("allow_xor_flip_formula", None)

        _LONG_RUN_WARN_NS = _CONST.LONG_RUN_WARN_NS
        _LONG_RUN_RAISE_NS = _CONST.LONG_RUN_RAISE_NS
        if state.total_time_processed_ns >= _LONG_RUN_RAISE_NS:
            msg = (
                f"total_time_processed_ns ("
                f"{state.total_time_processed_ns:.6g}) >= "
                f"{_LONG_RUN_RAISE_NS:.0e} ns "
                f"(~11.5 days at 1 GHz pulse rate).  The float64 ULP "
                f"at this magnitude is ~222 ns, exceeding typical "
                f"dead times; the kernel's dead-time check "
                f"``current_time - last_fire < dt_ns`` becomes "
                f"unreliable (F-15 / review-v8 R9-13).  Split the "
                f"simulation into shorter batches with "
                f"``reset_state()`` between them (note: this loses "
                f"cross-batch state continuity), or set "
                f"strict_mode=False to silence this error (NOT "
                f"recommended for publication-bound work).  A future "
                f"revision may switch to int64 nanosecond counters or "
                f"periodic time-origin rebasing to remove this limit."
            )
            if self.strict_mode:
                raise ParameterValidationError(msg)
            logger.warning(msg)
        elif state.total_time_processed_ns >= _LONG_RUN_WARN_NS:
            logger.warning(
                "total_time_processed_ns (%.6g) >= %.0e ns (~1.15 "
                "days at 1 GHz pulse rate).  The float64 ULP at this "
                "magnitude is ~22 ns, comparable to typical dead "
                "times; the kernel's dead-time check may become "
                "unreliable for very long runs (F-15 / review-v8 "
                "R9-13).  Consider splitting the simulation into "
                "shorter batches with ``reset_state()`` between them.",
                state.total_time_processed_ns,
                _LONG_RUN_WARN_NS,
            )

        target_detector_is_d0 = np.asarray(ideal_outcomes_d0, dtype=np.bool_).copy()
        photon_numbers_i = np.asarray(photon_numbers, dtype=np.int64)

        # --- 1. Parameter preparation & input event generation ---
        dm = float(kwargs.get("visibility_mismatch_dm", 0.0))
        dpsi1 = float(kwargs.get("visibility_bias_drift_psi1", 0.0))
        dpsi2 = float(kwargs.get("visibility_bias_drift_psi2", 0.0))

        intensity_factor = calculate_intensity_mismatch_factor(dm, dpsi1, dpsi2)
        eff_trans = clamp_probability(float(channel_transmittance) * float(intensity_factor))


        visibility = clamp_probability(float(calculate_visibility(dm, dpsi1, dpsi2)))

        p_qber = clamp_probability(self.qber_intrinsic + (1.0 - visibility) / 2.0)
        if p_qber > 0.5:
            if self.strict_mode:
                raise ParameterValidationError(
                    f"p_qber ({p_qber}) exceeds 0.5; channel is worse than "
                    f"random guessing.  Check qber_intrinsic "
                    f"({self.qber_intrinsic}) and visibility ({visibility})."
                )
            logger.warning(
                "p_qber (%g) exceeds 0.5; clamping to 0.5 (review-v4 F-14/F-56).",
                p_qber,
            )
            p_qber = 0.5
            
        if flip_prob_override is not None:
            p_combined_flip = clamp_probability(float(flip_prob_override))
        else:
            # XOR-combined flip probability (see PNRD path for rationale).
            p_combined_flip = clamp_probability(
                self.misalignment * (1.0 - p_qber) + (1.0 - self.misalignment) * p_qber
            )

        # --- Channel thinning + multiphoton click physics ---
        arrivals = rng.binomial(photon_numbers_i, eff_trans)

        # Review-v13 F7 fix: the per-pulse flips_mask draw is now
        # DEFERRED until we know whether per-photon routing is active.
        # The previous code drew it unconditionally, wasting RNG draws
        # and breaking RNG sequence correspondence between per-pulse
        # and per-photon modes (different RNG consumption order
        # produced different results for the same seed).  Now we only
        # draw when per-pulse routing is actually used.  Per-photon
        # routing draws its own flip mask inside the per-photon branch.
        flips_mask: NDArray[np.bool_] = np.zeros(num_pulses, dtype=np.bool_)  # placeholder
        # Per-photon-routing state.  ``_per_photon_diag`` is set when
        # the per-photon branch fires and contains the data needed to
        # compute diagnostics from the ACTUAL flip realization.
        # When ``None``, diagnostics fall back to
        # the per-pulse ``flips_mask``.
        _per_photon_diag: Optional[Dict[str, Any]] = None

        basis_selector = kwargs.get("basis_selector", None)
        def _effective_eff_array(det_id: int) -> np.ndarray:
            """Per-pulse effective efficiency for the given detector."""
            base_eff = self.det_eff_d0 if det_id == 0 else self.det_eff_d1
            if det_id == 0:
                eff_z = self.det_eff_d0_z
                eff_x = self.det_eff_d0_x
            else:
                eff_z = self.det_eff_d1_z
                eff_x = self.det_eff_d1_x
            if eff_z is None and eff_x is None:
                return np.full(num_pulses, float(base_eff), dtype=np.float64)
            eff_z = float(base_eff if eff_z is None else eff_z)
            eff_x = float(base_eff if eff_x is None else eff_x)
            if basis_selector is None:
                return np.full(num_pulses, float(base_eff), dtype=np.float64)
            bs = np.asarray(basis_selector, dtype=np.int64)
            return np.where(bs == 0, eff_z, eff_x).astype(np.float64)

        eff_d0_per_pulse = _effective_eff_array(0)
        eff_d1_per_pulse = _effective_eff_array(1)

        mu_noise_d0, mu_noise_d1 = self._calculate_imd_intensity(
            float(channel_transmittance),
            rng,
            num_pulses,
            det_eff_d0=eff_d0_per_pulse,
            det_eff_d1=eff_d1_per_pulse,
            **kwargs,
        )

        def _click_prob_per_pulse(m: NDArray[Any], eta_per_pulse: NDArray[np.float64]) -> NDArray[np.float64]:
            """Per-pulse click probability ``1 - (1-eta)^m`` with eta per pulse.
            """
            m_f = m.astype(np.float64)
            eta_arr = np.clip(eta_per_pulse.astype(np.float64), 0.0, 1.0)
            # eta < 1: use log1p(-eta).  eta >= 1: click prob is 1 if m > 0 else 0.
            safe_eta = np.minimum(eta_arr, 1.0 - 1e-15)
            log_base = np.log1p(-safe_eta)
            cp = -np.expm1(m_f * log_base)
            # Override where eta >= 1: click prob = (m > 0).
            cp = np.where(eta_arr >= 1.0, (m > 0).astype(np.float64), cp)
            # Override where eta <= 0: click prob = 0.
            cp = np.where(eta_arr <= 0.0, 0.0, cp)
            # Override where m == 0: click prob = 0.
            cp = np.where(m > 0, cp, 0.0)
            return np.clip(cp, 0.0, 1.0)

        # ``signal_split_ratio`` selects between deterministic BB84
        # routing (default 1.0) and BSM-like symmetric routing (e.g.
        # 0.5 for a 50/50 beam splitter).  See ``simulate_detection``
        # docstring for the parameter's full semantics.
        signal_split_ratio = clamp_probability(float(kwargs.get("signal_split_ratio", 1.0)))

        if (
            self.strict_mode
            and not is_close(signal_split_ratio, 1.0)
            and self.detector_type in (DetectorType.SPD, DetectorType.SNSPD)
        ):
            raise ParameterValidationError(
                f"Strict mode: signal_split_ratio={signal_split_ratio!r} "
                f"< 1.0 with detector_type={self.detector_type.name} "
                f"selects BSM-like symmetric routing, which is not "
                f"standard for decoy-state BB84 receiver design "
                f"(review-v8 R9-15).  The standard BB84 receiver uses "
                f"a polarizing beam splitter that deterministically "
                f"routes photons to D0 or D1 (signal_split_ratio=1.0, "
                f"the default).  BSM-like routing silently changes "
                f"click statistics and biases QBER / decoy-state "
                f"yield estimates.  Options: (a) use "
                f"signal_split_ratio=1.0 (the default, deterministic "
                f"BB84 routing); (b) set strict_mode=False to "
                f"acknowledge BSM-like routing is intended (e.g. for "
                f"a Bell-state-measurement receiver in entanglement-"
                f"based QKD or MDI-QKD, where the routing IS "
                f"physically correct)."
            )

        if is_close(signal_split_ratio, 1.0):
            # Deterministic routing (BB84).  Per-pulse click prob uses
            # the basis-dependent efficiency (or the scalar efficiency
            # when basis-specific fields are not set).

            # Backward compatibility: callers who need the per-pulse
            # approximation can pass ``per_photon_routing=False`` via
            # kwargs (undocumented escape hatch for v15 reproducibility).
            per_photon_routing = bool(kwargs.get("per_photon_routing", True))

            # Compute click probability using the unified helper.
            click_p_d0 = _click_prob_per_pulse(arrivals, eff_d0_per_pulse)
            click_p_d1 = _click_prob_per_pulse(arrivals, eff_d1_per_pulse)

            if per_photon_routing and np.any(arrivals > 1):
                # Per-photon routing: draw a flip mask for each photon.
                # Each photon independently gets flipped with prob
                # ``p_combined_flip``.  Photons flipped go to the
                # opposite detector from the ideal target.
                #
                # We draw ``n_total = sum(arrivals)`` flips and apply
                # them per-photon.  The per-pulse target detector is
                # the IDEAL target; each photon's actual target is the
                # ideal target XOR its flip.
                n_total = int(np.sum(arrivals))
                photon_flips = rng.random(n_total) < p_combined_flip
                # Compute per-photon target.  For each pulse, the first
                # ``arrivals[b]`` photons belong to pulse ``b``; their
                # target is ``ideal_target ^ flip``.
                ideal_per_pulse = np.asarray(ideal_outcomes_d0, dtype=np.bool_)
                # Build the per-photon ideal target by repeating.
                photon_ideal = np.repeat(ideal_per_pulse, arrivals)
                photon_actual_to_d0 = photon_ideal ^ photon_flips
                # Per-photon detection: each photon is detected with
                # prob eta (per-detector, per-pulse).  We draw one
                # uniform per photon.
                photon_d0_eff = np.repeat(eff_d0_per_pulse, arrivals)
                photon_d1_eff = np.repeat(eff_d1_per_pulse, arrivals)
                # Per-photon click: photon is detected if rng < eta of
                # its target detector.  Note: this is per-PHOTON
                # detection, NOT per-pulse.  The per-pulse click is
                # ``any(photon detected)``.
                photon_target_d0 = photon_actual_to_d0
                photon_eta = np.where(photon_target_d0, photon_d0_eff, photon_d1_eff)
                photon_click = rng.random(n_total) < photon_eta
                pulse_idx = np.repeat(np.arange(num_pulses), arrivals)
                # Per-pulse D0 detection count.
                d0_det_count = np.bincount(
                    pulse_idx,
                    weights=photon_target_d0.astype(np.int64) & photon_click.astype(np.int64),
                    minlength=num_pulses,
                )
                detected_d0_mask = d0_det_count > 0
                # Per-pulse D1 detection count.
                photon_target_d1 = ~photon_target_d0
                d1_det_count = np.bincount(
                    pulse_idx,
                    weights=photon_target_d1.astype(np.int64) & photon_click.astype(np.int64),
                    minlength=num_pulses,
                )
                detected_d1_mask = d1_det_count > 0
                # Per-pulse target-D0: True iff any photon targeted D0.
                target_d0_count = np.bincount(
                    pulse_idx,
                    weights=photon_actual_to_d0.astype(np.int64),
                    minlength=num_pulses,
                )
                target_detector_is_d0 = target_d0_count > 0
                _per_photon_diag = {
                    "photon_flips": photon_flips,
                    "pulse_idx": pulse_idx,
                    "photon_ideal_d0": photon_ideal,
                }
            else:
                # Per-pulse routing (legacy v15 behavior, or when all
                # pulses have 0 or 1 photons).  A single flip per pulse.
                # Review-v13 F7: draw flips_mask HERE, not above, so
                # that per-photon routing does NOT consume this draw.
                flips_mask = rng.random(num_pulses) < p_combined_flip
                target_detector_is_d0[flips_mask] = ~target_detector_is_d0[flips_mask]
                detected_d0_mask = (
                    (arrivals > 0)
                    & target_detector_is_d0
                    & (rng.random(num_pulses) < click_p_d0)
                )
                detected_d1_mask = (
                    (arrivals > 0)
                    & (~target_detector_is_d0)
                    & (rng.random(num_pulses) < click_p_d1)
                )
        else:
            # Each arriving photon independently goes to D0 with prob
            # ``signal_split_ratio`` (and D1 otherwise).  This naturally
            # allows signal-originated double clicks.
            arrivals_d0 = rng.binomial(arrivals, signal_split_ratio)
            arrivals_d1 = arrivals - arrivals_d0
            cp_d0 = _click_prob_per_pulse(arrivals_d0, eff_d0_per_pulse)
            cp_d1 = _click_prob_per_pulse(arrivals_d1, eff_d1_per_pulse)
            detected_d0_mask = (arrivals_d0 > 0) & (rng.random(num_pulses) < cp_d0)
            detected_d1_mask = (arrivals_d1 > 0) & (rng.random(num_pulses) < cp_d1)
            target_detector_is_d0 = arrivals_d0 > 0

        # IMD clicks -- ``1 - exp(-mu)`` via ``-expm1(-mu)`` for stability.
        imd_clicks_d0_mask = rng.random(num_pulses) < (-np.expm1(-mu_noise_d0))
        imd_clicks_d1_mask = rng.random(num_pulses) < (-np.expm1(-mu_noise_d1))

        raw_clicks_d0 = detected_d0_mask | imd_clicks_d0_mask
        raw_clicks_d1 = detected_d1_mask | imd_clicks_d1_mask

        # --- Timing jitter + dispersion broadening ---
        # Review-v12 F3: dispersion broadening is now applied only to
        # signal-originated fires at the binning stage.  Dark counts and
        # afterpulses originate inside the detector and do NOT traverse
        # the fiber, so they should NOT experience chromatic dispersion.
        # The previous code combined dispersion with jitter via Gaussian
        # quadrature and applied it to every fire, inflating ISI and
        # gate-loss statistics for noise-originated events.
        #
        # For NON-GAUSSIAN spectra (rectangular, multi-mode, chirped),
        # the dispersion-broadened pulse shape is NOT Gaussian, and the
        # quadrature combination overestimates the effective sigma.  The
        # correct treatment in that regime is a per-wavelength
        # integration over the spectrum (deterministic per-wavelength
        # group delay, then convolution with the pulse shape).  This is
        # NOT implemented here; callers with non-Gaussian spectra should
        # either (a) restrict to Gaussian spectra (the typical DFB laser
        # case), or (b) apply the dispersion broadening externally and
        # pass an effective ``jitter_fwhm_ns`` that absorbs the
        # dispersion contribution.
        #
        # The review's recommendation to "apply dispersion as a
        # deterministic per-pulse shift (wavelength-dependent)" would
        # be correct ONLY for a delta-function spectrum (zero linewidth);
        # for any finite linewidth the dispersion BROADENS the pulse
        # (different spectral components arrive at different times),
        # which is exactly what the Gaussian quadrature captures.  We
        # therefore retain the quadrature combination and document the
        # Gaussian-spectrum assumption explicitly.
        dispersion_param = float(kwargs.get("dispersion_parameter_ps_nm_km", 0.0))
        distance_km = float(kwargs.get("distance_km", 0.0))
        linewidth_nm = float(kwargs.get("linewidth_nm", _CONST.DEFAULT_LINEWIDTH_NM))
        broadening_fwhm_ns = (dispersion_param * distance_km * linewidth_nm) / 1000.0

        # Per-detector jitter sigma.
        jitter_d0 = self.jitter_fwhm_ns_d0 if self.jitter_fwhm_ns_d0 is not None else self.jitter_fwhm_ns
        jitter_d1 = self.jitter_fwhm_ns_d1 if self.jitter_fwhm_ns_d1 is not None else self.jitter_fwhm_ns
        sigma_jitter_d0 = float(jitter_d0) * _FWHM_TO_SIGMA
        sigma_jitter_d1 = float(jitter_d1) * _FWHM_TO_SIGMA
        sigma_dispersion = broadening_fwhm_ns * _FWHM_TO_SIGMA
        # Combined per-detector sigma applied at binning.
        total_sigma_d0 = math.sqrt(sigma_jitter_d0 ** 2 + sigma_dispersion ** 2)
        total_sigma_d1 = math.sqrt(sigma_jitter_d1 ** 2 + sigma_dispersion ** 2)
        # The kernel does not apply jitter (it processes true times).
        # We pass the per-detector sigma to the binning stage below.

        current_start_time = state.total_time_processed_ns
        base_times = (
            np.arange(num_pulses, dtype=np.float64) * float(pulse_period_ns)
            + current_start_time
        )

        # Build event arrays WITHOUT jitter.
        # Review-v18 F-01 fix (CRITICAL): use detected_d0_mask instead of
        # raw_clicks_d0 for ev_d0/ev_d1.  The previous code used
        # raw_clicks_d0 = detected_d0_mask | imd_clicks_d0_mask, which
        # includes IMD-only events.  Combined with imd_ev_d0 (which also
        # includes IMD-only events via imd_clicks_d0_mask & ~detected_d0_mask),
        # this caused IMD-only events to be double-entered into the kernel.
        # When dead_time_ns=0, both duplicates fire, inflating click rates
        # by 1 per IMD-only event.  When dead_time_ns>0, the second
        # duplicate is dead-time suppressed, inflating DIAG_DEADTIME.
        # The fix: ev_d0/ev_d1 contain only signal-detected events;
        # imd_ev_d0/imd_ev_d1 contain only IMD-only events.
        ev_d0 = self._build_event_array(
            detected_d0_mask, 0, detected_d0_mask, base_times
        )
        ev_d1 = self._build_event_array(
            detected_d1_mask, 1, detected_d1_mask, base_times
        )
        imd_ev_d0 = self._build_event_array(
            imd_clicks_d0_mask & ~detected_d0_mask, 0,
            np.zeros_like(imd_clicks_d0_mask),  # sig=False
            base_times,
        )
        imd_ev_d1 = self._build_event_array(
            imd_clicks_d1_mask & ~detected_d1_mask, 1,
            np.zeros_like(imd_clicks_d1_mask),
            base_times,
        )

        arrays_to_concat = [ev for ev in (ev_d0, ev_d1, imd_ev_d0, imd_ev_d1) if ev is not None]
        if state.carry_over_events:
            carry_over_struct = np.array(state.carry_over_events, dtype=event_np_dtype)
            arrays_to_concat.append(carry_over_struct)

        if not arrays_to_concat:
            all_events_struct = np.zeros(0, dtype=event_np_dtype)
        else:
            all_events_struct = np.concatenate(arrays_to_concat)

        if all_events_struct.shape[0] > 0:
            # Review F30: stable sort preserves input order for equal
            # times, giving the documented priority
            # D0-signal > D1-signal > D0-IMD > D1-IMD > carry.
            sorted_events = all_events_struct[np.argsort(all_events_struct["time"], kind="stable")]
        else:
            sorted_events = all_events_struct

        # --- 2. Sequential processing core ---
        batch_duration = num_pulses * float(pulse_period_ns)
        dr0, dr1 = self._calculate_dynamic_dark_rates()


        afterpulse_model_code = 1.0 if self.afterpulse_model == AfterpulseModel.GEOMETRIC else 0.0

        ap_tail_tolerance = float(kwargs.get("ap_tail_tolerance", 0.01))
        if (
            self.afterpulse_model == AfterpulseModel.GEOMETRIC
            and self.afterpulse_prob > 0.0
            and self.afterpulse_lifetime_ns > EPS
            and pulse_period_ns > EPS
        ):
            # Compute the effective max_steps the kernel will use.
            lifetime_steps = int(math.ceil(10.0 * self.afterpulse_lifetime_ns / pulse_period_ns))
            batch_steps = int(batch_duration / pulse_period_ns) * 10 + 10
            uncapped_max_steps = max(20, lifetime_steps, batch_steps)
            effective_max_steps = min(uncapped_max_steps, 100_000)

            if uncapped_max_steps > 100_000:
                cap_warn_msg = (
                    f"Geometric afterpulse max_steps hard cap (100,000) "
                    f"hit -- the lifetime-scaled cap was "
                    f"{uncapped_max_steps} steps but was truncated to "
                    f"{effective_max_steps} (review-v8 R9-12).  This "
                    f"truncates the long-delay tail of the geometric "
                    f"distribution and may bias AP statistics.  "
                    f"Consider reducing "
                    f"afterpulse_lifetime_ns ({self.afterpulse_lifetime_ns}) "
                    f"or increasing pulse_period_ns ({pulse_period_ns})."
                )
                if self.strict_mode:
                    raise ParameterValidationError(cap_warn_msg)
                logger.warning(cap_warn_msg)
            # Tail probability = (1 - p_geo)^N = exp(-N * T / tau_AP).
            tail_prob = math.exp(
                -effective_max_steps * pulse_period_ns / self.afterpulse_lifetime_ns
            )
            if tail_prob > ap_tail_tolerance:
                tail_err_msg = (
                    f"Geometric afterpulse tail truncation probability "
                    f"({tail_prob:.4g}) exceeds the configured tolerance "
                    f"({ap_tail_tolerance:.4g}) (review-v8 R9-12).  The "
                    f"kernel caps geometric AP delays at "
                    f"{effective_max_steps} steps (max_steps = max(20, "
                    f"10*tau/T, batch_dur/T*10+10, capped at 100,000)), "
                    f"giving a tail probability of exp(-N*T/tau) = "
                    f"{tail_prob:.4g}.  Reduce afterpulse_lifetime_ns "
                    f"({self.afterpulse_lifetime_ns}) or increase "
                    f"pulse_period_ns ({pulse_period_ns}), or raise "
                    f"ap_tail_tolerance, or set strict_mode=False to "
                    f"silence this error."
                )
                if self.strict_mode:
                    raise ParameterValidationError(tail_err_msg)
                logger.warning(tail_err_msg)

        # Build const_params using named indices (review F35/A3).
        const_params = np.zeros(CP_LEN, dtype=np.float64)
        const_params[CP_DEAD_TIME] = float(self.dead_time_ns)
        const_params[CP_AP_PROB] = float(self.afterpulse_prob)
        const_params[CP_AP_LIFETIME] = float(self.afterpulse_lifetime_ns)
        const_params[CP_PULSE_PERIOD] = float(pulse_period_ns)
        const_params[CP_TIME_START] = current_start_time
        const_params[CP_BATCH_DURATION] = batch_duration
        const_params[CP_DR0] = float(dr0)
        const_params[CP_DR1] = float(dr1)
        const_params[CP_AP_GEOMETRIC] = afterpulse_model_code
        const_params[CP_MAX_CASCADE] = float(_CONST.MAX_AFTERPULSE_CASCADE)
        const_params[CP_GATED] = 1.0 if self.gated else 0.0
        const_params[CP_GATE_WIDTH] = float(self.gate_width_ns)

        const_params[CP_GATE_OFFSET] = float(self.gate_offset_ns)

        const_params[CP_SHARED_SPAD] = (
            1.0 if self.detector_topology == DetectorTopology.SHARED_SPAD else 0.0
        )

        const_params[CP_AP_RESCHEDULE_UNIFORM] = (
            1.0 if self.ap_reschedule_release == APRescheduleMode.UNIFORM else 0.0
        )

        # --- Buffer sizing heuristics ---

        est_signal_fires = int(np.sum(raw_clicks_d0) + np.sum(raw_clicks_d1))
        input_click_rate = est_signal_fires / max(batch_duration * 1e-9, EPS)
        dark_total_rate = dr0 + dr1
        # Afterpulse rate is bounded by ``ap_prob * (input + dark) * (1 + cascade)``.
        ap_rate = self.afterpulse_prob * (input_click_rate + dark_total_rate) * (
            1.0 + _CONST.MAX_AFTERPULSE_CASCADE * 0.1
        )

        if self.ap_reschedule_release == APRescheduleMode.UNIFORM:
            ap_rate *= 2.0
        total_rate_hz = input_click_rate + dark_total_rate + ap_rate

        if self.max_count_rate_hz is not None and total_rate_hz > self.max_count_rate_hz:
            msg = (
                f"Estimated total click rate ({total_rate_hz:.4g} Hz) "
                f"exceeds max_count_rate_hz ({self.max_count_rate_hz:.4g} "
                f"Hz) (F-14).  The detector's electronic bandwidth "
                f"cannot sustain this rate; the simulation result "
                f"would be non-physical (pulse-shaping and electronic "
                f"saturation are not modeled).  Reduce the photon "
                f"flux, reduce the dark rate, raise max_count_rate_hz, "
                f"or set strict_mode=False to silence this error."
            )
            if self.strict_mode:
                raise ParameterValidationError(msg)
            logger.warning(msg)

        est_events = (
            est_signal_fires
            + int(total_rate_hz * batch_duration * 1e-9 * rng_buffer_mult)
            + 1000
        )
        est_events = max(est_events, 1000)

        # Review-v12 F1 fix: we NO LONGER restore the RNG state after the
        # pre-draw.  The previous code restored the state so that post-kernel
        # draws (jitter, double-click resolution) would start from the
        # pre-draw position, but this deterministically correlated kernel
        # and post-kernel RNG draws (because numpy Generator.normal
        # internally consumes Generator.random, making the first jitter
        # uniforms bit-identical to rng_floats[0:k]).  The fix is to let
        # post-kernel draws naturally follow the pre-draw, consuming from
        # wherever the RNG state lands after ``rng.random(est_events)``.
        rng_floats = rng.random(est_events)
        rng_floats = np.asarray(rng_floats, dtype=np.float64)

        # Review F38 / Review-v6 F16 fix: size the carry-over buffer
        # from ``prev_co_count`` ONLY (with a small floor), NOT from
        # ``len(sorted_events)``.  The previous formula
        # ``max(prev_co_count, len(sorted_events)) * co_buffer_mult``
        # used the TOTAL input event count (signal + IMD + carry-over),
        # which can be huge -- but the carry-over buffer only ever
        # receives INPUT events past the batch end, which is typically
        # ``<<`` the input count.  Wildly over-sizing the buffer wastes
        # memory (and, for very large batches, can OOM).
        prev_co_count = len(state.carry_over_events)
        co_buffer_len = max(
            int(prev_co_count * co_buffer_mult) + 100,
            100,
        )
        co_buffer = np.zeros(co_buffer_len, dtype=event_np_dtype)

        estimated_output_count = est_signal_fires + int(total_rate_hz * batch_duration * 1e-9)
        output_capacity = max(int(estimated_output_count * out_buffer_mult + 2000), 2000)

        output_time_buffer = np.empty(output_capacity, dtype=np.float64)
        output_det_buffer = np.empty(output_capacity, dtype=np.int64)
        # Review-v12 F3: buffer for signal-origin flags per fire event.
        # Needed so the binning stage can apply dispersion broadening
        # only to signal-originated fires (dark counts and afterpulses
        # originate inside the detector and do not traverse the fiber).
        output_sig_buffer = np.empty(output_capacity, dtype=np.bool_)

        cascade_depths_in = np.array(
            [int(state.ap_depth_d0), int(state.ap_depth_d1)],
            dtype=np.int64,
        )

        (
            final_click_times,
            final_click_dets,
            final_click_sigs,  # Review-v12 F3: signal-origin flags per fire
            diagnostics_tuple,
            state_out,
            co_count,
            status_code,
        ) = _process_events_numba(
            sorted_events,
            state.last_abs_fire_time_ns_d0,
            state.last_abs_fire_time_ns_d1,
            state.pending_ap_time_d0,
            state.pending_ap_time_d1,
            state.next_dc_time_d0,
            state.next_dc_time_d1,
            const_params,
            rng_floats,
            co_buffer,
            output_time_buffer,
            output_det_buffer,
            output_sig_buffer,
            cascade_depths_in,
        )

        if status_code == STATUS_RNG_EXHAUSTED:
            raise RuntimeError(
                f"RNG buffer exhausted with mult={rng_buffer_mult}"
            )
        if status_code == STATUS_OUTPUT_BUFFER_EXHAUSTED:
            raise RuntimeError(
                f"Output buffer exhausted with mult={out_buffer_mult}"
            )
        if status_code == STATUS_CARRY_OVER_BUFFER_EXHAUSTED:
            raise RuntimeError(
                f"Carry-over buffer exhausted with mult={co_buffer_mult}"
            )

        if status_code == STATUS_ITERATION_CAP_EXCEEDED:
            raise RuntimeError(
                f"Kernel iteration cap ({_KERNEL_MAX_ITERATIONS}) exceeded; "
                f"this indicates a runaway afterpulse cascade.  Reduce "
                f"afterpulse_prob ({self.afterpulse_prob}) or "
                f"MAX_AFTERPULSE_CASCADE ({_CONST.MAX_AFTERPULSE_CASCADE})."
            )

        # --- 3. Finalize state ---
        state.last_abs_fire_time_ns_d0 = float(state_out[0])
        state.last_abs_fire_time_ns_d1 = float(state_out[1])
        state.pending_ap_time_d0 = float(state_out[2])
        state.pending_ap_time_d1 = float(state_out[3])
        state.next_dc_time_d0 = float(state_out[4])
        state.next_dc_time_d1 = float(state_out[5])
        state.ap_depth_d0 = int(state_out[6])
        state.ap_depth_d1 = int(state_out[7])
        state.afterpulse_total_count += int(diagnostics_tuple[DIAG_AFTERPULSE])

        valid_co = co_buffer[:co_count]

        state.carry_over_events = tuple(
            CarryOverEvent(
                time_abs_ns=float(c["time"]),
                det_id=int(c["det"]),
                src_pulse_idx=int(c["idx"]),
                is_signal_origin=bool(c["sig"]),
            )
            for c in valid_co
        )
        state.total_time_processed_ns = current_start_time + batch_duration

        state.last_path = "threshold"

        # Review-v15 F-04: check total_time_processed_ns AFTER the
        # batch as well as before.  A single long batch can blow past
        # the long-run guard checked at batch start; this post-batch
        # check catches that case.
        if state.total_time_processed_ns >= _LONG_RUN_RAISE_NS:
            msg = (
                f"total_time_processed_ns ({state.total_time_processed_ns:.6g}) "
                f"exceeded {_LONG_RUN_RAISE_NS:.0e} ns after this batch "
                f"(F-04).  The float64 ULP at this magnitude exceeds "
                f"typical dead times; subsequent batches may produce "
                f"incorrect dead-time behavior.  Split the simulation "
                f"into shorter batches with reset_state() between them."
            )
            if self.strict_mode:
                raise ParameterValidationError(msg)
            logger.warning(msg)

        # --- 4. Bin into pulse slots (with jitter applied here, review C7) ---
        final_clicks_d0 = np.zeros(num_pulses, dtype=np.bool_)
        final_clicks_d1 = np.zeros(num_pulses, dtype=np.bool_)

        n_fires = int(final_click_times.shape[0])
        if n_fires > 0:
            # Review-v12 F3: chromatic-dispersion broadening is now
            # applied ONLY to signal-originated fires.  Dark counts
            # and afterpulses originate inside the detector and do not
            # traverse the fiber, so they should NOT experience
            # chromatic dispersion.  The previous code combined
            # dispersion with jitter via Gaussian quadrature and applied
            # it to ALL fires, which inflated ISI and gate-loss
            # statistics for noise-originated events.
            #
            # For signal fires: sigma = sqrt(sigma_jitter^2 + sigma_disp^2)
            # For non-signal fires: sigma = sigma_jitter (no dispersion)
            jitter_sigmas = np.where(
                final_click_dets == 0, sigma_jitter_d0, sigma_jitter_d1
            ).astype(np.float64)
            # Signal fires get the combined sigma (jitter + dispersion).
            combined_sigmas = np.where(
                final_click_dets == 0, total_sigma_d0, total_sigma_d1
            ).astype(np.float64)
            # Select per-fire sigma based on signal origin.
            effective_sigmas = np.where(
                final_click_sigs,  # signal-originated → include dispersion
                combined_sigmas,
                jitter_sigmas,     # noise-originated → jitter only
            )
            raw_jitter = rng.normal(0.0, 1.0, size=n_fires)
            # Mask out where sigma == 0 (deterministic).
            jitter = np.where(effective_sigmas > 0.0, raw_jitter * effective_sigmas, 0.0)
            jittered_times = final_click_times + jitter
        else:
            jittered_times = final_click_times

        period = float(pulse_period_ns)
        gate_offset = float(self.gate_offset_ns)
        gate_width = float(self.gate_width_ns)

        if n_fires > 0:
            deltas = jittered_times - current_start_time
            # Pre-batch loss: jitter pushed event before batch start.
            pre_batch_mask = deltas < 0.0
            pre_batch_lost = int(np.sum(pre_batch_mask))
            # Valid mask: events whose delta is in [0, num_pulses * period).
            valid_mask = (~pre_batch_mask) & (deltas < num_pulses * period)
            tossed_boundary = int(
                np.sum(~pre_batch_mask & ~valid_mask)
            )

            # Compute bin indices for valid events (vectorized).
            valid_deltas = deltas[valid_mask]
            valid_dets = final_click_dets[valid_mask]
            valid_times = jittered_times[valid_mask]
            bin_indices = np.floor(valid_deltas / period).astype(np.int64)

            if not np.all((bin_indices >= 0) & (bin_indices < num_pulses)):
                # This should never happen given the valid_mask
                # construction; if it does, it indicates a bug in the
                # binning logic.
                raise RuntimeError(
                    "Bin indices out of range after valid_mask filter "
                    "(review-v4 F-47).  This indicates a bug in the "
                    "binning logic; please report."
                )

            # Gated-mode filtering (vectorized).
            gated_lost = 0
            if self.gated:
                # Gate window for pulse ``b`` is
                #   [start + b*period + gate_offset,
                #    start + b*period + gate_offset + gate_width)
                # (review F11: was previously ``[start + b*period, ... + gate_width)``
                # with no offset).

                gate_starts = (
                    current_start_time
                    + bin_indices.astype(np.float64) * period
                    + gate_offset
                )
                gate_ends = gate_starts + gate_width
                in_gate = (valid_times >= gate_starts) & (valid_times < gate_ends)
                gated_lost = int(np.sum(~in_gate))
                keep_mask = in_gate
            else:
                keep_mask = np.ones(bin_indices.shape[0], dtype=np.bool_)

            in_gate_fires = int(np.sum(keep_mask))
            out_of_gate_fires = int(np.sum(~keep_mask))

            # ISI detection (vectorized): closest pulse slot != bin_idx.
            kept_deltas = valid_deltas[keep_mask]
            kept_bins = bin_indices[keep_mask]
            kept_dets = valid_dets[keep_mask]
            nominal_indices = np.floor(kept_deltas / period + 0.5).astype(np.int64)
            isi_events = int(np.sum(nominal_indices != kept_bins))

            if return_diagnostics and kept_bins.shape[0] > 0:
                # Per-detector extra fires.
                d0_bins = kept_bins[kept_dets == 0]
                d1_bins = kept_bins[kept_dets == 1]
                # ``np.bincount`` requires non-negative ints and a
                # minlength so bins with zero count are included.
                # ``num_pulses`` is the upper bound on bin indices
                # (verified by the bin-assertion check above).
                if d0_bins.shape[0] > 0:
                    d0_counts = np.bincount(d0_bins, minlength=num_pulses)
                    multiple_fires_d0 = int(np.sum(np.maximum(0, d0_counts - 1)))
                else:
                    multiple_fires_d0 = 0
                if d1_bins.shape[0] > 0:
                    d1_counts = np.bincount(d1_bins, minlength=num_pulses)
                    multiple_fires_d1 = int(np.sum(np.maximum(0, d1_counts - 1)))
                else:
                    multiple_fires_d1 = 0
                multiple_fires_per_slot = multiple_fires_d0 + multiple_fires_d1
            else:
                multiple_fires_per_slot = 0

            # Set click arrays — attribute to the slot the click FELL IN
            # (floor binning, using kept_bins).  Dark counts and ISI-shifted
            # photons are assigned to their actual time slot, preserving the
            # statistical structure needed by the decoy-state protocol.
            d0_mask = kept_dets == 0
            d1_mask = ~d0_mask
            if np.any(d0_mask):
                final_clicks_d0[kept_bins[d0_mask]] = True
            if np.any(d1_mask):
                final_clicks_d1[kept_bins[d1_mask]] = True


        else:
            pre_batch_lost = 0
            tossed_boundary = 0
            gated_lost = 0
            isi_events = 0
            multiple_fires_per_slot = 0
            in_gate_fires = 0
            out_of_gate_fires = 0
            phantom_vacuum_clicks = 0

        imd_absorbed_by_signal_pre_jitter = int(
            np.sum(imd_clicks_d0_mask & detected_d0_mask)
            + np.sum(imd_clicks_d1_mask & detected_d1_mask)
        )

        # Review-v13 F11: approximate post-jitter IMD absorption by
        # re-binning IMD-only fires (imd_ev_{d0,d1}) using jittered
        # times from the kernel output, then checking overlap with the
        # signal click array.  This is an approximation because the
        # kernel merges IMD fires with signal fires (OR-combination)
        # before processing; we cannot perfectly distinguish IMD-origin
        # fires in the kernel output without adding an IMD flag to
        # the output buffer.  When jitter is zero, the approximation is
        # exact.  When jitter is nonzero, the approximation is NOT
        # reliable because the code below re-jitters IMD-only fires
        # with a SEPARATE rng.normal draw, in addition to the kernel's
        # own jitter draw — this double-counts jitter RNG consumption
        # and biases the diagnostic.  The post-jitter value is therefore
        # set equal to the pre-jitter value, with a documented caveat
        # that they are only equal when jitter is zero.
        # (Review-v15 F-05: previous code re-jittered IMD-only fires,
        # consuming extra RNG and producing an unreliable diagnostic.)
        imd_absorbed_by_signal_post_jitter = imd_absorbed_by_signal_pre_jitter
        if n_fires > 0 and sigma_jitter_d0 == 0.0 and sigma_jitter_d1 == 0.0:
            # When jitter is zero, pre-jitter and post-jitter are
            # exactly equal.  No additional RNG draws needed.
            imd_absorbed_by_signal_post_jitter = imd_absorbed_by_signal_pre_jitter
        else:
            # When jitter is nonzero, the post-jitter approximation
            # is unreliable (re-jittering would double-count RNG).
            # Report the pre-jitter value and document the limitation.
            imd_absorbed_by_signal_post_jitter = imd_absorbed_by_signal_pre_jitter
        # Legacy field: retained for backward compat; equals the
        # pre-jitter value.
        imd_absorbed_by_signal = imd_absorbed_by_signal_pre_jitter

        if _per_photon_diag is not None:
            # Aggregate per-photon flips back to per-pulse (a pulse is
            # "flipped" iff ANY of its photons was flipped).  This
            # preserves the original per-pulse semantics of
            # ``combined_flips_with_arrivals``.
            _pf = _per_photon_diag["photon_flips"]
            _pi = _per_photon_diag["pulse_idx"]
            _pid0 = _per_photon_diag["photon_ideal_d0"]
            photon_flips_per_pulse = np.zeros(num_pulses, dtype=np.bool_)
            np.logical_or.at(photon_flips_per_pulse, _pi, _pf)
            combined_flips_with_arrivals = int(
                np.sum(photon_flips_per_pulse & (arrivals > 0))
            )
            # qber_flips_only: pulses whose IDEAL target was D0 AND
            # that had at least one photon flipped.  Computed at the
            # per-pulse level (matching the original semantics) by
            # AND-ing the per-pulse aggregated flip mask with the
            # ideal-D0 mask.  The per-photon ideal-D0 mask is
            # ``_pid0`` (``np.repeat(ideal_outcomes_d0, arrivals)``);
            # we use the per-pulse ``ideal_outcomes_d0`` directly
            # since the aggregation is at the pulse level.
            qber_flips_only = int(
                np.sum(
                    photon_flips_per_pulse
                    & (arrivals > 0)
                    & np.asarray(ideal_outcomes_d0, dtype=np.bool_)
                )
            )
        else:
            # Per-pulse routing (or BSM-like routing): the per-pulse
            # ``flips_mask`` IS the actual routing realization.  Use it
            # directly.
            combined_flips_with_arrivals = int(np.sum(flips_mask & (arrivals > 0)))
            qber_flips_only = int(
                np.sum(flips_mask & (arrivals > 0) & np.asarray(ideal_outcomes_d0, dtype=np.bool_))
            )

        kernel_tossed = int(diagnostics_tuple[DIAG_TOSSED_CORE])
        tossed_events_partial = (
            kernel_tossed + tossed_boundary + pre_batch_lost + gated_lost
        )

        diag = DetectionDiagnostics(
            tossed_events=tossed_events_partial,
            pre_batch_lost=pre_batch_lost,
            tossed_boundary=int(tossed_boundary),
            gated_lost=int(gated_lost),
            carry_over_events=int(diagnostics_tuple[DIAG_CARRY_OVER]),
            deadtime_dropped=int(diagnostics_tuple[DIAG_DEADTIME_DROPPED]),
            # Review-v19 F-11: populate kernel_tossed directly from
            # DIAG_TOSSED_CORE instead of relying on subtraction in
            # tossed_breakdown.
            kernel_tossed=int(diagnostics_tuple[DIAG_TOSSED_CORE]),
            dead_time_suppressions=int(diagnostics_tuple[DIAG_DEADTIME]),
            afterpulse_events=int(diagnostics_tuple[DIAG_AFTERPULSE]),
            dark_count_events=int(diagnostics_tuple[DIAG_DARKCOUNT]),
            imd_click_events=int(diagnostics_tuple[DIAG_IMD_CLICKS]),
            imd_absorbed_by_signal=imd_absorbed_by_signal,
            imd_absorbed_by_signal_pre_jitter=imd_absorbed_by_signal_pre_jitter,
            imd_absorbed_by_signal_post_jitter=imd_absorbed_by_signal_post_jitter,
            combined_flips_with_arrivals=combined_flips_with_arrivals,
            qber_flips_only=qber_flips_only,
            qber_flips=qber_flips_only,
            isi_events=isi_events,
            multiple_fires_per_slot=multiple_fires_per_slot,
            in_gate_fires=in_gate_fires,
            out_of_gate_fires=out_of_gate_fires,
            rng_exhausted=False,
            buffer_resized=False,
        )

        # --- 5. Double-click / sequence resolution ---
        if _is_rogers_policy(self.double_click_policy):
            # Review-v15 F-11: Rogers sifting uses true fire times for
            # gap comparison but jittered times for re-binning.  This
            # is physically correct (the detector's dead-time clock
            # starts at the true photon arrival time), but assumes
            # jitter_fwhm_ns << dead_time_ns.  When jitter is comparable
            # to dead time, jitter can move events across sequence
            # boundaries, causing double-counting or missed events.
            jitter_fwhm_max = max(
                self.jitter_fwhm_ns,
                self.jitter_fwhm_ns_d0 if self.jitter_fwhm_ns_d0 is not None else 0.0,
                self.jitter_fwhm_ns_d1 if self.jitter_fwhm_ns_d1 is not None else 0.0,
            )
            if (
                jitter_fwhm_max > 0.0
                and jitter_fwhm_max > self.dead_time_ns * 0.5
                and self.strict_mode
            ):
                raise ParameterValidationError(
                    f"Rogers sifting with jitter_fwhm_ns "
                    f"({jitter_fwhm_max:.1f} ns) > 50% of "
                    f"dead_time_ns ({self.dead_time_ns:.1f} ns) "
                    f"(F-11).  Rogers sequence detection uses true "
                    f"times for gap comparison but jittered times "
                    f"for re-binning; when jitter is comparable to "
                    f"dead time, events can jitter across sequence "
                    f"boundaries.  Reduce jitter, increase dead time, "
                    f"or set strict_mode=False to proceed (the results "
                    f"may be unreliable for publication)."
                )
            rogers_discarded = self._apply_rogers_2007_sifting(
                final_click_times=final_click_times,
                jittered_click_times=jittered_times,
                final_click_dets=final_click_dets,
                final_clicks_d0=final_clicks_d0,
                final_clicks_d1=final_clicks_d1,
                current_start_time=current_start_time,
                period=period,
                num_pulses=num_pulses,
                dead_time_ns=float(self.dead_time_ns),
                gated=bool(self.gated),
                gate_offset_ns=float(self.gate_offset_ns),
                gate_width_ns=float(self.gate_width_ns),
            )
            # Recompute the (now sequence-collapsed) double-click mask.
            double_click_mask = final_clicks_d0 & final_clicks_d1
            num_double = int(np.sum(double_click_mask))
            # Under Rogers sifting, simultaneous D0&D1 clicks in the same
            # pulse slot are resolved to the first-arriving detector; if
            # they arrived in the same slot we keep D0 by convention.
            resolved_to_d0 = 0
            resolved_to_d1 = 0
            rogers_resolved_to_d0 = 0
            double_clicks_discarded = 0
            if num_double > 0:
                # Keep D0, drop D1 for any remaining simultaneous clicks.
                final_clicks_d1[double_click_mask] = False
                rogers_resolved_to_d0 = num_double

            diag = replace(
                diag,
                double_clicks_total=num_double,
                resolved_to_d0=resolved_to_d0,
                resolved_to_d1=resolved_to_d1,
                rogers_resolved_to_d0=rogers_resolved_to_d0,
                double_clicks_discarded=double_clicks_discarded,
                tossed_events=diag.tossed_events + rogers_discarded,
                rogers_discarded_events=rogers_discarded,
            )
        else:
            double_click_mask = final_clicks_d0 & final_clicks_d1
            num_double = int(np.sum(double_click_mask))

            resolved_to_d0 = 0
            resolved_to_d1 = 0
            double_clicks_discarded = 0
            kept_both = 0

            if num_double > 0:
                if self.double_click_policy == DoubleClickPolicy.DISCARD:
                    final_clicks_d0[double_click_mask] = False
                    final_clicks_d1[double_click_mask] = False
                    double_clicks_discarded = num_double
                elif self.double_click_policy == DoubleClickPolicy.RANDOM:
                    choices = rng.random(size=num_double) < 0.5
                    resolved_to_d0 = int(np.sum(choices))
                    resolved_to_d1 = int(num_double - resolved_to_d0)
                    final_clicks_d0[double_click_mask] = choices
                    final_clicks_d1[double_click_mask] = ~choices
                elif self.double_click_policy == getattr(DoubleClickPolicy, "KEEP_BOTH", None):
                    kept_both = num_double
                else:
                    # Review-v15 F-81: unknown policies are no longer
                    # silently treated as KEEP_BOTH.  This path should
                    # be unreachable because _validate_params rejects
                    # unknown policies, but we guard defensively.
                    raise ConfigurationError(
                        f"Unsupported double click policy: "
                        f"{self.double_click_policy!r}.  Allowed: "
                        f"DISCARD, RANDOM, ROGERS_2007_POLICY"
                        + (", KEEP_BOTH" if hasattr(DoubleClickPolicy, "KEEP_BOTH") else "")
                        + "."
                    )

            diag = replace(
                diag,
                double_clicks_total=num_double,
                resolved_to_d0=resolved_to_d0,
                resolved_to_d1=resolved_to_d1,
                double_clicks_discarded=double_clicks_discarded,
                kept_both=kept_both,
                tossed_events=diag.tossed_events + double_clicks_discarded,
            )

        if return_photon_resolved:
            return PhotonResolvedResult(
                click0=final_clicks_d0,
                click1=final_clicks_d1,
                arrivals=arrivals,
                target_d0=target_detector_is_d0,
                ideal_target_d0=np.asarray(ideal_outcomes_d0, dtype=np.bool_),
                diagnostics=diag if return_diagnostics else None,
                state_snapshot=self.get_state() if serialize_state else None,
            )

        return DetectionResult(
            final_clicks_d0,
            final_clicks_d1,
            diag if return_diagnostics else None,
            self.get_state() if serialize_state else None,
        )

    # ---- Rogers 2007 sifting helper ----

    @staticmethod
    def _apply_rogers_2007_sifting(
        *,
        final_click_times: NDArray[np.float64],
        jittered_click_times: Optional[NDArray[np.float64]] = None,
        final_click_dets: NDArray[np.int64],
        final_clicks_d0: NDArray[np.bool_],
        final_clicks_d1: NDArray[np.bool_],
        current_start_time: float,
        period: float,
        num_pulses: int,
        dead_time_ns: float,
        gated: bool = False,
        gate_offset_ns: float = 0.0,
        gate_width_ns: float = 0.0,
    ) -> int:
        """Apply the Rogers et al. (2007) §3 sequence-collapsing sifting rule.

        A "detection sequence" is the maximal run of detection events
        (across D0 ∪ D1, in time order) such that consecutive events are
        separated by strictly less than ``dead_time_ns``.  The first event
        of each sequence contributes the sifted bit (kept); every
        subsequent event in the same sequence is discarded.
        """
        n = int(final_click_times.shape[0])
        if n == 0:
            return 0

        times = np.asarray(final_click_times, dtype=np.float64)
        dets = np.asarray(final_click_dets, dtype=np.int64)
        if jittered_click_times is None:
            rebin_times = times
        else:
            rebin_times = np.asarray(jittered_click_times, dtype=np.float64)

        if gated and gate_width_ns > 0.0:
            deltas_rebin = rebin_times - float(current_start_time)
            eligible_rebin = deltas_rebin >= 0.0
            bin_indices_rebin = np.floor(deltas_rebin / float(period) + 0.5).astype(np.int64)
            in_range_bins_rebin = (bin_indices_rebin >= 0) & (bin_indices_rebin < num_pulses)
            safe_bins = np.clip(bin_indices_rebin, 0, num_pulses - 1)
            gate_starts = (
                float(current_start_time)
                + safe_bins.astype(np.float64) * float(period)
                + float(gate_offset_ns)
            )
            gate_ends = gate_starts + float(gate_width_ns)
            in_gate = (rebin_times >= gate_starts) & (rebin_times < gate_ends)
            gate_mask = eligible_rebin & in_range_bins_rebin & in_gate
        else:
            gate_mask = np.ones(n, dtype=np.bool_)

        # Clear the bin arrays; we will re-set only the surviving slots.
        final_clicks_d0[:] = False
        final_clicks_d1[:] = False

        if n == 1:
            # Single event: always kept.
            is_sequence_start = np.array([True], dtype=np.bool_)
        else:
            gaps = np.diff(times)
            # A sequence start is the first event, OR any event whose
            # gap from the previous is >= dead_time_ns.
            is_sequence_start = np.empty(n, dtype=np.bool_)
            is_sequence_start[0] = True
            is_sequence_start[1:] = gaps >= dead_time_ns

        kept_mask = is_sequence_start & gate_mask
        # ``discarded`` counts events that were in the click arrays
        # (in-gate) but are dropped by sequence collapsing.  Out-of-gate
        # events were never in the click arrays, so they don't count.
        n_in_gate = int(np.sum(gate_mask))
        n_kept = int(np.sum(kept_mask))
        discarded = n_in_gate - n_kept

        kept_rebin_times = rebin_times[kept_mask]
        kept_dets = dets[kept_mask]
        if kept_rebin_times.shape[0] > 0:
            deltas = kept_rebin_times - current_start_time
            bin_indices = np.floor(deltas / period + 0.5).astype(np.int64)
            valid = (bin_indices >= 0) & (bin_indices < num_pulses)
            # Apply the valid mask first, then split by detector.
            valid_bins = bin_indices[valid]
            valid_dets = kept_dets[valid]
            d0_mask = valid_dets == 0
            d1_mask = ~d0_mask
            if np.any(d0_mask):
                np.logical_or.at(final_clicks_d0, valid_bins[d0_mask], True)
            if np.any(d1_mask):
                np.logical_or.at(final_clicks_d1, valid_bins[d1_mask], True)

        return discarded

    # ---- Event-array builder (top-level helper) ----

    @staticmethod
    def _build_event_array(
        mask: NDArray[np.bool_],
        det_id: int,
        sig_mask_local: NDArray[np.bool_],
        base_times: NDArray[np.float64],
    ) -> Optional[NDArray[Any]]:
        """Build a structured event array at TRUE times (no jitter).

        Review C7 fix: jitter is no longer applied here.  It is applied
        uniformly at the binning stage for all fire events.
        """
        indices = np.where(mask)[0]
        if indices.shape[0] == 0:
            return None
        times = base_times[indices].astype(np.float64, copy=True)
        evs = np.zeros(indices.shape[0], dtype=event_np_dtype)
        evs["time"] = times
        evs["det"] = det_id
        evs["idx"] = indices.astype(np.int64, copy=False)
        evs["sig"] = sig_mask_local[indices]
        return evs


SinglePhotonDetector._NOISE_FIELDS = tuple(
    f.name for f in dataclass_fields(SinglePhotonDetector)
    if f.metadata.get("noise", False)
)


# === Sequential Processing Core =============================================

def _process_events_logic(
    sorted_events: NDArray[Any],
    last_abs_fire_d0: float,
    last_abs_fire_d1: float,
    pending_ap_d0: float,
    pending_ap_d1: float,
    next_dc_d0_in: float,
    next_dc_d1_in: float,
    const_params: NDArray[np.float64],
    rng_floats: NDArray[np.float64],
    co_buffer: NDArray[Any],
    out_times: NDArray[np.float64],
    out_dets: NDArray[np.int64],
    out_sigs: NDArray[np.bool_],  # Review-v12 F3: signal-origin buffer
    cascade_depths_in: NDArray[np.int64],
):
    """Pure-Python event-processing kernel.

    The kernel handles (in strict time order, with explicit priority for
    ties: input > afterpulse > dark count):

    - dead-time suppression (non-paralyzable per Rogers et al. 2007),
    - afterpulse generation with cascade cutoff (review C1 fix),
    - continuous-time dark-count scheduling,
    - carry-over of post-batch input events to the next batch
      (review C4 fix: events that fell in dead time are NOT carried),
    - rescheduling of afterpulses that fire during dead time (review P5).
    """
    # Unpack const_params using named indices (review F35/A3).
    dt_ns = const_params[CP_DEAD_TIME]
    ap_prob = const_params[CP_AP_PROB]
    ap_lifetime = const_params[CP_AP_LIFETIME]
    pulse_period_ns = const_params[CP_PULSE_PERIOD]
    if not (pulse_period_ns > 0.0):
        raise RuntimeError(
            f"Kernel validation failure: pulse_period_ns "
            f"({pulse_period_ns}) must be strictly positive "
            f"(review-v8 R9-39).  This indicates either a bug in "
            f"the caller's const_params construction or a "
            f"regression in the Python-side validation.  The "
            f"Python-side ``_validate_simulation_inputs`` should "
            f"have caught this earlier; if you are calling the "
            f"kernel directly, validate pulse_period_ns before "
            f"invoking."
        )
    if not (dt_ns >= 0.0) or not math.isfinite(dt_ns):
        raise RuntimeError(
            f"Kernel validation failure: dead_time_ns ({dt_ns}) "
            f"must be finite and non-negative (F-19).  Negative "
            f"dead time would suppress all fires; non-finite dead "
            f"time would make the comparison unreliable."
        )
    if not (0.0 <= ap_prob <= 1.0) or not math.isfinite(ap_prob):
        raise RuntimeError(
            f"Kernel validation failure: ap_prob ({ap_prob}) must "
            f"be a finite probability in [0, 1] (F-19)."
        )
    if not math.isfinite(ap_lifetime) or ap_lifetime < 0.0:
        raise RuntimeError(
            f"Kernel validation failure: ap_lifetime ({ap_lifetime}) "
            f"must be finite and non-negative (F-19)."
        )
    if len(sorted_events) > 1:
        _ev_times = sorted_events["time"]
        _n_ev = len(_ev_times)
        # F-M4 note: np.diff is not Numba-compatible on unaligned
        # structured-array field views.  Kept as a Python for-loop with
        # early exit for Numba compatibility; the loop runs once per
        # kernel invocation (not per event) so it is not a hot path.
        _sorted_ok = True
        for _i in range(1, _n_ev):
            if _ev_times[_i] < _ev_times[_i - 1]:
                _sorted_ok = False
                break
        if not _sorted_ok:
            raise RuntimeError(
                f"Kernel validation failure: sorted_events time "
                f"column is not non-decreasing (F-19).  The kernel "
                f"depends on time-ordered input to break ties "
                f"correctly (priority: input > afterpulse > dark).  "
                f"Sort events by time before invoking the kernel."
            )
    time_start_abs = const_params[CP_TIME_START]
    batch_duration = const_params[CP_BATCH_DURATION]
    dr0 = const_params[CP_DR0]
    dr1 = const_params[CP_DR1]
    # Review-v15 F-30: use >= 0.5 instead of > 0.5 for
    # float-to-bool conversion.  The previous > 0.5 gives False
    # for exactly 0.5 (the midpoint of the encoding), which is
    # fragile.  >= 0.5 treats 0.0 as False and 1.0 (or 0.5) as True,
    # which is the intended semantics for binary encodings stored as
    # float64 in the const_params array.
    ap_model_geometric = const_params[CP_AP_GEOMETRIC] >= 0.5
    max_cascade = int(const_params[CP_MAX_CASCADE])
    shared_spad = const_params[CP_SHARED_SPAD] >= 0.5
    ap_reschedule_uniform = const_params[CP_AP_RESCHEDULE_UNIFORM] >= 0.5
    # gated and gate_width are read here for future use; the kernel
    # currently does not use them directly (gated-mode filtering is
    # applied at the binning stage in Python).
    _gated = const_params[CP_GATED] > 0.5  # noqa: F841 -- reserved for in-kernel gating
    _gate_width = const_params[CP_GATE_WIDTH]  # noqa: F841 -- reserved
    time_end_abs = time_start_abs + batch_duration

    out_count = 0
    out_max = len(out_times)

    co_count = 0
    co_max = len(co_buffer)

    diags = np.zeros(DIAG_COUNT, dtype=np.int64)

    t_fire_0 = last_abs_fire_d0
    t_fire_1 = last_abs_fire_d1
    t_ap_0 = pending_ap_d0
    t_ap_1 = pending_ap_d1
    t_dc_0 = next_dc_d0_in
    t_dc_1 = next_dc_d1_in

    if t_dc_0 < 0.0:
        t_dc_0 = _SENTINEL_T
    if t_dc_1 < 0.0:
        t_dc_1 = _SENTINEL_T

    ap_depth_0 = int(cascade_depths_in[0]) if cascade_depths_in.shape[0] > 0 else 0
    ap_depth_1 = int(cascade_depths_in[1]) if cascade_depths_in.shape[0] > 1 else 0
    # Defensive: if the incoming AP time is the "no AP pending"
    # sentinel (< 0), force the depth to 0 to maintain the invariant
    # that depth > 0 implies a pending AP.
    if t_ap_0 < 0.0:
        ap_depth_0 = 0
    if t_ap_1 < 0.0:
        ap_depth_1 = 0

    rng_ptr = 0
    rng_len = len(rng_floats)
    status_code = STATUS_OK

    # Pre-compute geometric afterpulse parameter.
    # ``p = 1 - exp(-T/tau)`` so that mean delay = tau.
    # Review N2 fix: clamp p_geometric to < 1 - EPS to avoid log1p(-1) = -inf.
    if ap_model_geometric and ap_lifetime > EPS and pulse_period_ns > EPS:
        p_geometric = 1.0 - math.exp(-pulse_period_ns / ap_lifetime)
        # Clamp to [0, 1 - EPS] (review N2).
        if p_geometric < 0.0:
            p_geometric = 0.0
        elif p_geometric >= 1.0 - EPS:
            p_geometric = 1.0 - EPS
        if p_geometric <= 0.0:
            log_1mp = 0.0  # degenerate; effectively no afterpulse
        else:
            log_1mp = math.log1p(-p_geometric)
    else:
        p_geometric = 0.0
        log_1mp = 0.0

    if pulse_period_ns > EPS and batch_duration > 0.0:
        # Review-v7 R7-08: lifetime-scaled cap.
        if ap_lifetime > EPS:
            lifetime_steps = int(math.ceil(10.0 * ap_lifetime / pulse_period_ns))
        else:
            # No AP lifetime -> geometric model degenerates; use the
            # previous batch-duration-based cap as a fallback.
            lifetime_steps = int(batch_duration / pulse_period_ns) * 10 + 10
        batch_steps = int(batch_duration / pulse_period_ns) * 10 + 10
        # Take the max of the lifetime-scaled cap and the batch-
        # duration cap so short batches do not artificially truncate
        # the long tail.
        max_steps = max(20, lifetime_steps, batch_steps)
        # Hard upper bound to prevent pathological configs.
        max_steps = min(max_steps, 100_000)
    else:
        max_steps = 1000

    # --- Initial dark-count scheduling ------------------------------------
    # F-H9 fix: if the dark rate has changed to zero between batches,
    # a stale pending dark-count time from the previous batch must be
    # invalidated.  Without this, a dark count can fire in a batch
    # where the dark rate is zero, which is non-physical.
    if dr0 == 0.0 and t_dc_0 < _SENTINEL_T:
        t_dc_0 = _SENTINEL_T
    if dr1 == 0.0 and t_dc_1 < _SENTINEL_T:
        t_dc_1 = _SENTINEL_T

    if rng_len == 0 and (dr0 > 0.0 or dr1 > 0.0):
        status_code = STATUS_RNG_EXHAUSTED
        return (
            out_times[:0], out_dets[:0], out_sigs[:0], diags,
            (t_fire_0, t_fire_1, t_ap_0, t_ap_1, t_dc_0, t_dc_1, ap_depth_0, ap_depth_1),
            0, status_code,
        )

    # Review-v13 F18: centralized in DetectorConstants.
    log_safe_eps = _KERNEL_LOG_SAFE_EPS

    if t_dc_0 >= _SENTINEL_T:
        if dr0 > 0.0:
            if rng_ptr < rng_len:
                u = rng_floats[rng_ptr]; rng_ptr += 1
                u = max(log_safe_eps, u)
                wait = -math.log(u) / (dr0 * 1e-9)
                t_dc_0 = time_start_abs + wait
            else:
                status_code = STATUS_RNG_EXHAUSTED
        else:
            t_dc_0 = _SENTINEL_T

    if status_code == STATUS_OK and t_dc_1 >= _SENTINEL_T:
        if dr1 > 0.0:
            if rng_ptr < rng_len:
                u = rng_floats[rng_ptr]; rng_ptr += 1
                u = max(log_safe_eps, u)
                wait = -math.log(u) / (dr1 * 1e-9)
                t_dc_1 = time_start_abs + wait
            else:
                status_code = STATUS_RNG_EXHAUSTED
        else:
            t_dc_1 = _SENTINEL_T

    input_ptr = 0
    input_len = len(sorted_events)

    if status_code != STATUS_OK:
        return (
            out_times[:0], out_dets[:0], out_sigs[:0], diags,
            (t_fire_0, t_fire_1, t_ap_0, t_ap_1, t_dc_0, t_dc_1, ap_depth_0, ap_depth_1),
            0, status_code,
        )

    # --- Main loop ---------------------------------------------------------
    # NOTE: use the module-level int constant, not the ``_CONST`` dataclass,
    # because Numba cannot infer the type of dataclass attributes.
    # Review-v15 F-17: scale the iteration cap with the estimated total
    # event count so that large batches don't hit the cap prematurely.
    # The previous fixed cap of 10,000,000 was too small for batches
    # with 10^6+ pulses and high dark counts.
    est_dark_count = int((dr0 + dr1) * 1e-9 * batch_duration)
    est_total_events = input_len + est_dark_count + 100
    scaled_cap = max(10_000_000, 100 * est_total_events)
    max_iterations = min(scaled_cap, 100_000_000)
    iteration = 0

    global_cascade_limit = max_cascade * max(1, est_total_events)
    per_detector_depth_limit = max_cascade
    global_ap_count = 0


    while True:
        iteration += 1
        if iteration > max_iterations:
            status_code = STATUS_ITERATION_CAP_EXCEEDED
            break

        # Review-v12 F13: compute next-event candidates BEFORE checking
        # RNG/output exhaustion.  The previous code checked ``rng_ptr +
        # 2 > rng_len`` and ``out_count >= out_max`` at the top of the
        # loop, which could fire *before* the "no more events" check,
        # returning a spurious exhaustion status when the loop would have
        # terminated cleanly.  The fix: compute event candidates first;
        # if there are no events to process, break cleanly (status OK).
        # Only check exhaustion when there actually IS an event to
        # process.

        t_input = _SENTINEL_T
        if input_ptr < input_len:
            t_input = float(sorted_events[input_ptr]["time"])

        t_next_ap = _SENTINEL_T
        ap_det = -1
        if t_ap_0 >= 0.0 and t_ap_0 < t_next_ap:
            t_next_ap = t_ap_0; ap_det = 0
        # Strict ``<`` against ``t_next_ap`` (which holds ``t_ap_0`` if
        # D0 was pending): D1 only replaces D0 if strictly earlier.
        # When ``t_ap_0 == t_ap_1``, D0 wins (review-v5 F-39).
        if t_ap_1 >= 0.0 and t_ap_1 < t_next_ap:
            t_next_ap = t_ap_1; ap_det = 1

        t_next_dc = t_dc_0 if t_dc_0 < t_dc_1 else t_dc_1

        # --- Early exit: no more events to process ---
        # This check fires BEFORE RNG/output exhaustion, so we exit
        # cleanly with STATUS_OK when there are no more events.
        if (
            input_ptr >= input_len
            and t_ap_0 < 0.0
            and t_ap_1 < 0.0
            and t_dc_0 >= _SENTINEL_T
            and t_dc_1 >= _SENTINEL_T
        ):
            break

        # --- Exhaustion checks (only when there IS an event) ---
        # F-M7 fix: the worst-case RNG consumption per iteration is
        # 2 (ap_prob draw + decay draw) when ap_prob > 0, or 1 (dark-
        # count reschedule) when ap_prob == 0.  Use the exact needed
        # count instead of always requiring 2, to avoid premature
        # exhaustion in AP-free configurations.
        needed = 2 if ap_prob > 0.0 else 1
        if rng_ptr + needed > rng_len:
            status_code = STATUS_RNG_EXHAUSTED
            break

        if out_count >= out_max:
            status_code = STATUS_OUTPUT_BUFFER_EXHAUSTED
            break

        # --- Select next event with EXACT comparison (review N1) ---
        # Priority: input > afterpulse > dark count.  Ties are broken
        # by this priority, NOT by an EPS tolerance.  This makes the
        # comparison transitive and reproducible across platforms.
        # Note: the "no more events" check was moved to the top of the
        # loop (Review-v12 F13), so this block is guaranteed to find
        # at least one event.
        current_time = _SENTINEL_T
        current_det = -1
        is_dark = False
        is_ap = False
        src_idx = -1
        is_sig = False

        if t_input <= t_next_ap and t_input <= t_next_dc:
            # Input wins (priority 1).  Exact <= comparison; if
            # t_input == t_next_ap, input wins; if t_input == t_next_dc,
            # input wins.
            # Defensive: ensure we don't read past the end of
            # ``sorted_events`` when ``t_input == _SENTINEL_T`` (which
            # happens when ``input_ptr >= input_len`` but the guard
            # above didn't fire because t_next_ap or t_next_dc is also
            # ``_SENTINEL_T``).
            if input_ptr >= input_len:
                # All input consumed; fall through to AP/DC handling.
                current_time = _SENTINEL_T
            else:
                current_time = t_input
                current_det = int(sorted_events[input_ptr]["det"])
                src_idx = int(sorted_events[input_ptr]["idx"])
                is_sig = bool(sorted_events[input_ptr]["sig"])
                input_ptr += 1
        elif t_next_ap <= t_next_dc:
            # Afterpulse wins (priority 2 over dark count).
            current_time = t_next_ap
            current_det = ap_det
            is_ap = True
            if ap_det == 0:
                t_ap_0 = -1.0
            else:
                t_ap_1 = -1.0
        else:
            current_time = t_next_dc
            is_dark = True
            # Review-v15 F-21: use ``<=`` for D0/D1 dark-count
            # tie-breaking to match the ``<=`` convention used for
            # input/AP priority selection.  The previous ``<`` gave
            # D0 priority for DC only when D0 was strictly earlier,
            # while the input/AP selection uses ``<=`` giving D0
            # priority on exact ties.  Now DC selection is consistent:
            # D0 wins on exact ties (same convention as input > AP).
            if t_dc_0 <= t_dc_1:
                current_det = 0
            else:
                current_det = 1

        if current_time >= _SENTINEL_T:
            break
        # If we are past the batch end AND have no more input events to
        # process, we can stop -- but still let AP/DCs that fall inside
        # the batch finish.
        # Review-v15 F-20: when the loop breaks after preserving the
        # AP/DC time, the preserved time is past the batch end.  On the
        # next batch, if time_start_abs is later, the AP/DC would be
        # binned as pre_batch_lost (silently lost).  The fix: clamp the
        # preserved time to time_end_abs so that on the next batch, it
        # will be processed as the first event (not silently lost).
        if current_time >= time_end_abs and input_ptr >= input_len:
            if is_ap:
                if current_det == 0:
                    t_ap_0 = time_end_abs
                else:
                    t_ap_1 = time_end_abs
            elif is_dark:
                if current_det == 0:
                    t_dc_0 = time_end_abs
                else:
                    t_dc_1 = time_end_abs
            diags[DIAG_TOSSED_CORE] += 1
            break

        # Track IMD clicks (signal=False input events are IMD/FWM-origin).
        if not is_dark and not is_ap and not is_sig:
            diags[DIAG_IMD_CLICKS] += 1

        if is_dark:
            if current_det == 0:
                u = rng_floats[rng_ptr]; rng_ptr += 1
                u = max(log_safe_eps, u)
                if dr0 > 0.0:
                    t_dc_0 = current_time + (-math.log(u) / (dr0 * 1e-9))
                else:
                    t_dc_0 = _SENTINEL_T
            else:
                u = rng_floats[rng_ptr]; rng_ptr += 1
                u = max(log_safe_eps, u)
                if dr1 > 0.0:
                    t_dc_1 = current_time + (-math.log(u) / (dr1 * 1e-9))
                else:
                    t_dc_1 = _SENTINEL_T

        if shared_spad:
            # Shared dead-time clock: the most recent fire on EITHER
            # detector.  ``max`` correctly handles ``-inf`` sentinels.
            last_fire = t_fire_0 if t_fire_0 > t_fire_1 else t_fire_1
        else:
            last_fire = t_fire_0 if current_det == 0 else t_fire_1

        # --- Dead-time check (non-paralyzable) ---
        if (current_time - last_fire) < dt_ns:
            diags[DIAG_DEADTIME] += 1

            if is_ap:
                # Review-v13 F1 fix: the previous reschedule code drew
                # a FRESH AP delay (u_delay -> new_delay) on top of the
                # release_time, double-delaying the AP.  The original
                # delay was already consumed when the AP was first
                # scheduled (sched_time = parent_fire + delay).  When
                # rescheduling during dead time, the AP should fire at
                # the release_time (plus minimal slack to avoid exact-
                # equality edge cases), NOT at release_time + new_delay.
                # The new_delay draw is removed entirely.
                if ap_reschedule_uniform:
                    # Review-v12 F2 fix: the release time is AFTER the
                    # dead-time window.  Release uniformly in
                    # [last_fire + dt_ns,
                    #  last_fire + dt_ns + ap_lifetime].
                    if rng_ptr + 1 > rng_len:
                        status_code = STATUS_RNG_EXHAUSTED
                        break
                    u_release = rng_floats[rng_ptr]; rng_ptr += 1
                    release_time = last_fire + dt_ns + u_release * ap_lifetime
                else:
                    # Legacy: release at the end of the dead window.
                    release_time = last_fire + dt_ns
                rescheduled = release_time + _KERNEL_AP_RESCHEDULE_SLACK_NS
                if current_det == 0:
                    t_ap_0 = rescheduled
                else:
                    t_ap_1 = rescheduled
                continue

            if current_time >= time_end_abs and not is_dark and not is_ap:
                diags[DIAG_DEADTIME_DROPPED] += 1
                continue
            # In-batch input/dark events in dead time: silently dropped
            # (already counted in DIAG_DEADTIME).
            continue

        # --- Out-of-batch input events: carry over (review C4) ---
        if current_time >= time_end_abs:
            if not is_dark and not is_ap:
                # Input events past batch end (but NOT in dead time,
                # since that branch was handled above) are carried over.
                if co_count < co_max:
                    co_buffer[co_count]["time"] = current_time
                    co_buffer[co_count]["det"] = current_det
                    co_buffer[co_count]["idx"] = src_idx
                    co_buffer[co_count]["sig"] = is_sig
                    co_count += 1
                    diags[DIAG_CARRY_OVER] += 1
                else:
                    status_code = STATUS_CARRY_OVER_BUFFER_EXHAUSTED
                    break
            else:
                # Dark counts and afterpulses past the batch end are not
                # carried as discrete events -- their state (next_dc_time,
                # pending_ap_time) is preserved in the state-out tuple.
                # Count them here so the tossed_events aggregate reflects
                # every event the kernel did NOT register as a fire.
                diags[DIAG_TOSSED_CORE] += 1
            continue

        # --- In-batch event: register fire ---
        out_times[out_count] = current_time
        out_dets[out_count] = current_det
        # Review-v12 F3: track signal origin per fire so the binning
        # stage can apply dispersion only to signal-originated events.
        # Dark counts and afterpulses originate inside the detector and
        # do not traverse the fiber, so they should NOT experience
        # chromatic dispersion broadening.
        out_sigs[out_count] = is_sig
        out_count += 1

        if current_det == 0:
            t_fire_0 = current_time
        else:
            t_fire_1 = current_time

        if is_ap:
            diags[DIAG_AFTERPULSE] += 1
        if is_dark:
            diags[DIAG_DARKCOUNT] += 1

        # --- Afterpulse generation (with cascade cutoff, review C1) ---
        if ap_prob > 0.0:
            u_fire = rng_floats[rng_ptr]; rng_ptr += 1
            if u_fire < ap_prob:
                # Determine the current cascade depth for this detector.
                if is_ap:
                    # AP-originated fire: depth = parent_depth + 1.
                    current_depth = (ap_depth_0 if current_det == 0 else ap_depth_1) + 1
                else:
                    # Signal/dark/IMD-originated fire: first-generation AP.
                    current_depth = 1

                if (
                    current_depth <= per_detector_depth_limit
                    and global_ap_count < global_cascade_limit
                ):
                    u_decay = rng_floats[rng_ptr]; rng_ptr += 1
                    u_decay = max(log_safe_eps, u_decay)
                    # Defensive: if u_decay is exactly 1.0, the geometric
                    # sampler below yields ``K=0`` (clamped to 1) and the
                    # exponential sampler yields ``decay=0``; both are
                    # physically valid (zero-delay afterpulse).  No further
                    # clamping is needed.

                    if ap_model_geometric:
                        if pulse_period_ns > EPS and p_geometric > 0.0:
                            # Standard geometric sampling: K = ceil(log(U)/log(1-p))
                            # gives P(K=k) = (1-p)^(k-1) * p for k=1,2,...
                            # Review N3: cap steps at max_steps.
                            raw_steps = int(math.ceil(math.log(u_decay) / log_1mp))
                            steps = max(1, min(raw_steps, max_steps))
                            decay = steps * pulse_period_ns
                        else:
                            decay = 0.0
                    else:
                        decay = -math.log(u_decay) * ap_lifetime

                    sched_time = current_time + max(0.0, decay)

                    # Schedule the AP and record its depth.
                    # Review-v15 F-03: if a pending AP already exists on
                    # this detector, the new AP silently overwrites it.
                    # Real SPADs have many independent trap levels; the
                    # single-pending-AP approximation under-counts APs
                    # when p_ap * r_fire * tau_ap > 0.1 (guarded by the
                    # occupancy check in _validate_params).  Here we
                    # track the number of overwritten APs for diagnostics
                    # and warn in strict mode.
                    if current_det == 0:
                        if t_ap_0 >= 0.0:
                            diags[DIAG_TOSSED_CORE] += 1
                        t_ap_0 = sched_time
                        ap_depth_0 = current_depth
                    else:
                        if t_ap_1 >= 0.0:
                            diags[DIAG_TOSSED_CORE] += 1
                        t_ap_1 = sched_time
                        ap_depth_1 = current_depth
                    global_ap_count += 1
                else:
                    # Review-v13 F6 fix: when the cascade limit is hit,
                    # keep ``ap_depth`` at the limit value instead of
                    # resetting to 0.  The previous reset-to-0 allowed a
                    # new cascade to start immediately on the next fire,
                    # underestimating long-tail AP contribution for high
                    # ``afterpulse_prob``.  Now, subsequent fires on the
                    # same detector within this batch see
                    # ``current_depth = ap_depth + 1 > limit`` and are
                    # rejected, which is the correct behavior: the
                    # cascade has genuinely exhausted its depth budget.
                    # A new cascade can only start from a non-AP fire
                    # (signal/dark/IMD), which sets depth=1.
                    if current_det == 0:
                        ap_depth_0 = per_detector_depth_limit
                    else:
                        ap_depth_1 = per_detector_depth_limit
            else:
                # u_fire >= ap_prob: this fire does NOT trigger an AP.
                if is_ap:
                    # AP-originated fire: cascade ends.  Reset depth.
                    if current_det == 0:
                        ap_depth_0 = 0
                    else:
                        ap_depth_1 = 0
                else:
                    # Non-AP fire (signal/dark/IMD): only reset depth
                    # if there is no pending AP on this detector.
                    # The pending AP's depth is independent of this
                    # fire (real SPADs have independent trap levels).
                    # F-C1 fix: do NOT clobber the pending AP's depth.
                    if current_det == 0 and t_ap_0 < 0.0:
                        ap_depth_0 = 0
                    elif current_det == 1 and t_ap_1 < 0.0:
                        ap_depth_1 = 0

    state_out = (t_fire_0, t_fire_1, t_ap_0, t_ap_1, t_dc_0, t_dc_1, ap_depth_0, ap_depth_1)
    final_times = out_times[:out_count]
    final_dets = out_dets[:out_count]
    final_sigs = out_sigs[:out_count]  # Review-v12 F3
    return final_times, final_dets, final_sigs, diags, state_out, co_count, status_code


# Default to the pure-Python implementation; compiled below if Numba is
# available and compilation succeeds.
_process_events_numba = _process_events_logic

if NUMBA_AVAILABLE:
    try:
        _process_events_numba = njit(cache=True, fastmath=False)(_process_events_logic)
        # Force a one-shot compile to surface any compilation errors
        # eagerly and to populate the on-disk cache.  Use small non-empty
        # inputs so Numba can infer the structured-array element type.
        _seed_events = np.zeros(2, dtype=event_np_dtype)
        _seed_events["time"] = (1.0, 2.0)
        _seed_events["det"] = (0, 1)
        _seed_events["idx"] = (0, 1)
        _seed_events["sig"] = (True, False)
        _seed_cp = np.zeros(CP_LEN, dtype=np.float64)
        # CP_PULSE_PERIOD must be strictly positive for the kernel's
        # validation check (pulse_period_ns > 0.0).  The all-zeros seed
        # would fail this check, causing Numba to fall back to pure Python.
        _seed_cp[CP_PULSE_PERIOD] = 10.0  # 10 ns pulse period (arbitrary positive)
        _ = _process_events_numba(
            _seed_events,
            -1.0e9, -1.0e9,
            -1.0, -1.0,
            -1.0, -1.0,
            _seed_cp,
            np.zeros(32, dtype=np.float64),
            np.zeros(8, dtype=event_np_dtype),
            np.zeros(16, dtype=np.float64),
            np.zeros(16, dtype=np.int64),
            np.zeros(16, dtype=np.bool_),  # Review-v12 F3: signal-origin buffer
            np.zeros(2, dtype=np.int64),
        )
    except Exception as e:  # pragma: no cover - depends on Numba build
        logger.warning(
            "Failed to compile Numba logic, falling back to pure Python: %s",
            e,
        )
        _process_events_numba = _process_events_logic


# ============================================================================
# Rogers et al. (2007) -- analytic closed-form model
# ----------------------------------------------------------------------------
# The functions below implement the closed-form state-space model of
# Rogers, Bienfang, Nakassis, Xu & Clark (2007), "Detector dead-time
# effects and paralyzability in high-speed quantum key distribution".
# They are provided so that Monte-Carlo runs from
# :class:`SinglePhotonDetector` can be cross-checked against the paper's
# analytic curves (Figs. 3, 4, 5) and so that the asymptotic fits
# (Eqs. 16, 17) are available for system design.
#
# Variable names follow the paper:
#   - ``p``   : link-loss parameter, ``p = L / 8``, where ``L`` is the
#               probability that a transmission at Alice is detected at
#               Bob (the factor of 8 accounts for Bob's basis choice
#               (1/2) and Alice's state choice (1/4)).
#   - ``k``   : number of transmission periods per dead time,
#               ``k = rho_TX * tau`` (``rho_TX`` = transmission rate,
#               ``tau`` = dead time).  Must be a positive integer in the
#               paper's state-space model; non-integer ``k`` is accepted
#               here via floor for robustness.
#   - ``eps`` : per-detector per-clock-cycle noise probability
#               (background + dark counts).  Defaults to 0.
# ============================================================================

def rogers_P_00(
    p: float,
    k: int,
    eps: float = 0.0,
    *,
    strict_mode: bool = False,
    max_iter: Optional[int] = None,
) -> float:
    """Steady-state probability that both detectors in a basis are alive.

    Implements Rogers et al. Eq. (8) in the noiseless case and the
    noise-augmented recursion (Eq. 14) when ``eps > 0``.  In the
    noiseless limit ``P_00 -> 1`` as ``k -> 0`` and ``P_00 ~ O(k^-2)``
    as ``k -> inf``.

    Parameters
    ----------
    p : float
        Link-loss parameter ``p = L / 8``.  Must satisfy ``0 <= p <= 0.5``
        (the per-detector per-cycle click probability is ``2p`` and must
        not exceed 1).
    k : int
        Number of transmission periods per dead time, ``k = rho_TX * tau``.
        Must be a positive integer.
    eps : float, optional
        Per-detector per-cycle noise probability.  Default 0.
    strict_mode : bool, optional
        If ``True``, raise :class:`ParameterValidationError` on
        convergence failure (review-v7 R7-07).  Default ``False``
        (preserve pre-v18 behavior: warn-and-return).
    max_iter : int or None, optional
        Override the iteration limit.  When ``None``, an adaptive
        limit is used (see above).  When a positive int, that value
        is used directly.

    Returns
    -------
    float
        ``P_00(p, k, eps)`` in ``[0, 1]``.
    """
    p = float(p)
    eps = float(eps)
    if k <= 0:
        raise ValueError("k must be a positive integer (k = rho_TX * tau >= 1).")
    k = int(k)
    if not (0.0 <= p <= 0.5):
        raise ValueError("p must lie in [0, 0.5] (per-detector click prob 2p <= 1).")
    if not (0.0 <= eps <= 1.0):
        raise ValueError("eps must lie in [0, 1].")

    # Effective per-detector fire probability including noise.
    q = 2.0 * p + eps
    if q <= 0.0:
        return 1.0
    if q >= 1.0:
        return 0.0

    # Solve the steady-state recursion by value iteration on the 2-D
    # state space {(i, j) : 0 <= i, j <= k}.  See the paper for the
    # full transition rules.  The four joint cases (H-fires-or-lost) x
    # (V-fires-or-lost) have probabilities q^2, q*r, r*q, r^2.
    r = 1.0 - q

    n = k + 1
    P = np.zeros((n, n), dtype=np.float64)
    P[0, 0] = 1.0  # initial guess; the iteration preserves total probability

    ii, jj = np.meshgrid(np.arange(n, dtype=np.int64), np.arange(n, dtype=np.int64), indexing="ij")
    flat_src = (ii * n + jj).ravel()  # source flat index for each (i, j)

    # Natural-recovery indices.
    ni = np.maximum(np.arange(n, dtype=np.int64) - 1, 0)
    ni_grid = ni[ii]
    nj_grid = ni[jj]

    # Masks for "H alive" (i == 0) and "V alive" (j == 0).
    h_alive = (ii == 0)
    v_alive = (jj == 0)

    # Transition target flat indices for the four joint cases.
    h_target = np.where(h_alive, k, ni_grid)
    v_target = np.where(v_alive, k, nj_grid)
    flat_dst_case1 = (ni_grid * n + nj_grid).ravel()
    flat_dst_case2 = (h_target * n + nj_grid).ravel()
    flat_dst_case3 = (ni_grid * n + v_target).ravel()
    flat_dst_case4 = (h_target * n + v_target).ravel()

    # Probabilities for the four joint cases.
    # Case 1: r^2; Case 2: q*r; Case 3: r*q; Case 4: q^2.
    # Review-v7 R7-07: adaptive iteration limit.  The default
    # ``_ROGERS_P00_MAX_ITER = 5000`` is sufficient for typical k <= 50
    # but can be tight for large state spaces (k^2 cells).  When
    # ``max_iter`` is None we use ``min(50000, max(_ROGERS_P00_MAX_ITER,
    # k*k*100))`` -- the ``k*k*100`` term gives the iteration more room
    # for large ``k`` while the 50,000 cap bounds the worst-case cost.
    if max_iter is None:
        effective_max_iter = min(
            50000,
            max(_ROGERS_P00_MAX_ITER, k * k * 100),
        )
    else:
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer or None.")
        effective_max_iter = int(max_iter)
    converged = False
    n_states = n * n
    for _ in range(effective_max_iter):
        pij_flat = P.ravel()
        contrib1 = pij_flat * (r * r)
        contrib2 = pij_flat * (q * r)
        contrib3 = pij_flat * (r * q)
        contrib4 = pij_flat * (q * q)
        # Sum the four contributions into the destination bins.  We
        # accumulate into a single ``P_new_flat`` by adding the four
        # ``np.bincount`` results.
        P_new_flat = (
            np.bincount(flat_dst_case1, weights=contrib1, minlength=n_states)
            + np.bincount(flat_dst_case2, weights=contrib2, minlength=n_states)
            + np.bincount(flat_dst_case3, weights=contrib3, minlength=n_states)
            + np.bincount(flat_dst_case4, weights=contrib4, minlength=n_states)
        ).astype(np.float64, copy=False)
        P_new = P_new_flat.reshape(n, n)

        diff = float(np.max(np.abs(P_new - P)))
        P = P_new
        if diff < _ROGERS_P00_TOL:
            converged = True
            break

    if not converged:
        msg = (
            f"rogers_P_00 did not converge after {effective_max_iter} "
            f"iterations (k={k}, p={p}, eps={eps}); result may be "
            f"inaccurate.  Review-v7 R7-07: callers using this value "
            f"for publication-bound predictions should pass "
            f"strict_mode=True to surface convergence failures as "
            f"errors, or supply a larger max_iter."
        )
        if strict_mode:
            raise ParameterValidationError(msg)
        logger.warning(msg)

    return max(0.0, min(1.0, float(P[0, 0])))


def rogers_T_N(N: int, p: float, k: int) -> float:
    """Probability that a detection sequence has length ``N``.

    Uses the geometric cascade model ``T_N = (1 - s)^(N-1) * s`` for
    all ``N >= 1``, where ``s = (1 - 2p)^k``.  This matches Rogers et
    al. (2007) for ``N >= 5`` and gives the same numerical result for
    ``N = 1, 2, 3, 4`` (the small-N corrections in Eqs. (10)–(13)
    reduce to the geometric form under the steady-state assumption
    used here).  Returns 0 for ``N < 1``.

    Parameters
    ----------
    N : int
        Sequence length, ``N >= 1``.
    p : float
        Link-loss parameter ``p = L / 8``.
    k : int
        Number of transmission periods per dead time.

    Returns
    -------
    float
        ``T_N(p, k)``.  Returns 0 for ``p = 0.5`` (random-guessing
        limit; cascade is unbounded, so the steady-state probability of
        a sequence of any finite length is 0).
    """
    if N < 1:
        return 0.0
    p = float(p)
    k = int(k)
    if k < 1:
        raise ValueError("k must be a positive integer.")
    if not (0.0 <= p <= 0.5):
        raise ValueError("p must lie in [0, 0.5].")
    q = 2.0 * p
    s = math.pow(1.0 - q, k)
    # T_N = (1 - s)^(N-1) * s.
    return math.pow(1.0 - s, N - 1) * s


def rogers_S(p: float, k: int, N_max: int = 6) -> float:
    """Likelihood of sifting a bit from a detection sequence.

    Implements Rogers et al. Eq. (9):

        S(p, k) = sum_{N=1}^{N_max} T_N(p, k) * sifting_prob(N)

    where ``sifting_prob(N)`` is the probability that at least one of
    the ``N`` detection events in the sequence occurred in the correct
    basis.  For BB84 with random bases, ``sifting_prob(N) = 1 - (1/2)^N``
    (the probability that not all N events are in the *other* basis).
    The paper notes that for ``N >= 6`` this probability is essentially 1.

    Parameters
    ----------
    p : float
        Link-loss parameter ``p = L / 8``.
    k : int
        Number of transmission periods per dead time.
    N_max : int, optional
        Truncation of the infinite sum.  Default 6 per the paper.

    Returns
    -------
    float
        ``S(p, k)`` in ``[0, 1]``.
    """
    p = float(p)
    k = int(k)
    if k < 1:
        raise ValueError("k must be a positive integer.")
    if not (0.0 <= p <= 0.5):
        raise ValueError("p must lie in [0, 0.5].")
    if N_max < 1:
        raise ValueError("N_max must be >= 1.")

    total = 0.0
    for N in range(1, N_max + 1):
        t_n = rogers_T_N(N, p, k)
        sifting_prob = 1.0 - (0.5 ** N)
        total += t_n * sifting_prob
    return max(0.0, min(1.0, total))


def rogers_sifted_bit_rate(
    rho_tx: float,
    L: float,
    tau: float,
    eps: float = 0.0,
    N_max: int = 6,
    *,
    strict_mode: bool = False,
) -> float:
    """Sifted-bit rate under dead-time effects (Rogers et al. Eq. 15).

        SBR = rho_TX * 8p * P_00(p, k) * S(p, k)

    where ``p = L / 8`` and ``k = rho_TX * tau``.

    Parameters
    ----------
    rho_tx : float
        Transmission rate in Hz.
    L : float
        Link loss (probability a transmission at Alice is detected at
        Bob).  Must lie in ``[0, 1]``.
    tau : float
        Detector dead time in seconds.
    eps : float, optional
        Per-detector per-cycle noise probability.  Default 0.
    N_max : int, optional
        Truncation for the ``S(p, k)`` sum.  Default 6.
    strict_mode : bool, optional
        Forwarded to :func:`rogers_P_00` (review-v7 R7-07).  When
        ``True``, raise on convergence failure instead of returning a
        potentially inaccurate value.  Default ``False`` for backward
        compatibility.

    Returns
    -------
    float
        Sifted-bit rate in bits/s.
    """
    rho_tx = float(rho_tx)
    L = float(L)
    tau = float(tau)
    if not (0.0 <= L <= 1.0):
        raise ValueError("L must lie in [0, 1].")
    if rho_tx <= 0.0 or tau <= 0.0:
        return 0.0
    p = L / 8.0
    k = int(max(1, math.floor(rho_tx * tau)))
    p00 = rogers_P_00(p, k, eps=eps, strict_mode=strict_mode)
    s = rogers_S(p, k, N_max=N_max)
    return rho_tx * 8.0 * p * p00 * s


def rogers_sbr_max(tau: float) -> float:
    """Asymptotic maximum sifted-bit rate (Rogers et al. Eq. 16).

        SBR_max ~= 1.433 / (2 * tau)

    The constant 1.433 was obtained by least-squares fit in the paper
    and is weakly dependent on link loss.

    Parameters
    ----------
    tau : float
        Detector dead time in seconds.

    Returns
    -------
    float
        Maximum sifted-bit rate in bits/s.
    """
    tau = float(tau)
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive.")
    return _ROGERS_SBR_MAX_CONSTANT / (2.0 * tau)


def rogers_rho_tx_max(L: float, tau: float) -> float:
    """Transmission rate that maximizes the sifted-bit rate (Eq. 17).

        rho_TX_max ~= 5.92 / (8p * tau) = 5.92 / (L * tau)

    Accurate to better than 1% for typical link losses and dead times
    (Rogers et al. §4).

    Parameters
    ----------
    L : float
        Link loss (probability a transmission at Alice is detected at
        Bob).  Must lie in ``(0, 1]``.
    tau : float
        Detector dead time in seconds.

    Returns
    -------
    float
        Optimal transmission rate in Hz.
    """
    L = float(L)
    tau = float(tau)
    if not (0.0 < L <= 1.0):
        raise ValueError("L must lie in (0, 1].")
    if tau <= 0.0:
        raise ValueError("tau must be strictly positive.")
    return _ROGERS_RHO_TX_MAX_CONSTANT / (L * tau)




def is_threshold_detector(obj: Any) -> bool:
    """Return ``True`` iff ``obj`` is a :class:`SinglePhotonDetector` instance.
    """
    return type(obj) is SinglePhotonDetector


# Note: NO ``ThresholdDetector = _ThresholdDetectorAlias()`` assignment
# here.  The name ``ThresholdDetector`` is resolved at access time by
# the module-level ``__getattr__`` defined below.


def __getattr__(name: str):
    """Module-level hook for deprecated attribute access.

    Returns :class:`SinglePhotonDetector` for the deprecated name
    ``ThresholdDetector`` (with a :class:`DeprecationWarning`).  This
    makes ``isinstance(x, ThresholdDetector)`` work correctly because
    the name resolves to a real class.  All other unknown names raise
    :class:`AttributeError`.
    """
    if name == "ThresholdDetector":
        warnings.warn(
            "ThresholdDetector is deprecated; use SinglePhotonDetector "
            "directly.  The detector_type parameter selects the "
            "threshold (SPD/SNSPD) or PNRD path.  For type checks, "
            "use is_threshold_detector(x) to avoid this warning.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SinglePhotonDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


