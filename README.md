 # QKD Simulation Framework — Decoy-State BB84 Finite-Key Security Analysis
 
 **Version:** 4.3.0 &nbsp;|&nbsp; **Language:** Python 3.10+ &nbsp;|&nbsp; **License:** Proprietary / Research
 
 ## Executive Summary
 
 This repository provides a modular, event-driven simulation framework for
 Quantum Key Distribution (QKD) protocols.  It targets finite-key decoy-state
 BB84 and Measurement-Device-Independent (MDI) QKD, with a focus on
 **physically realistic detector models**, **rigorous security proofs**, and
 **parameter-optimisation sweeps** for experimental design.
 
 The framework models the full QKD pipeline: photon-number statistics of the
 source (`qkd/sources.py`), scalar attenuation of the optical fibre channel
 (`qkd/channel.py`), stateful event-driven single-photon detectors with dead
 time, afterpulsing, dark counts, and timing jitter (`qkd/detectors.py`),
 protocol-level state preparation, measurement, and sifting
 (`qkd/protocols.py`), and finite-key security proofs (`qkd/proofs/`) drawn
 from the peer-reviewed literature (Lim _et al._ 2014, Ma _et al._ 2005,
 Wang 2005, MDI-QKD, and tight bounds).  A top-level runner
 (`main_optimized.py`) orchestrates parallel parameter sweeps, and a 7-node
 network simulator (`qkd_network_7nodes.py`) models trusted-relay QKD
 networks with SPF+ path selection.
 
 ## Repository Architecture
 
 ```
 .
 ├── qkd/                          # Core package
 │   ├── __init__.py               # Lazy-loading PEP 562 public API
 │   ├── datatypes.py              # Enums, dataclasses, SimulationResults
 │   ├── constants.py              # Physical constants, numeric tolerances
 │   ├── constants_definitions.py  # Constant definitions
 │   ├── constants_helpers.py      # Probability clamping, dB conversion
 │   ├── exceptions.py             # Hierarchical exception tree (>=6 types)
 │   ├── params.py                 # QKDParams: validation, serialization
 │   ├── type_defs.py              # RNG protocol type
 │   ├── sources.py                # PoissonSource, DensityMatrixSource
 │   ├── channel.py                # FiberChannel (Beer-Lambert attenuation)
 │   ├── detectors.py              # SPD/SNSPD/PNRD event-driven model
 │   ├── modulators.py             # Mach-Zehnder modulator (MZMConfig)
 │   ├── noise_models.py           # Thermal + shot electrical noise
 │   ├── protocols.py              # BB84, MDI-QKD, B92, Redundant
 │   ├── io.py                     # Atomic JSON save, gzip, SHA256
 │   ├── security_metadata.py      # SourceSecurityMetadata
 │   ├── scm_shim.py               # SCM backward-compatibility
 │   ├── proofs/                   # Security proof sub-package
 │   │   ├── base.py               # FiniteKeyProof ABC
 │   │   ├── lim2014.py            # Lim et al. (2014) finite-key
 │   │   ├── tight.py              # BB84 tight finite-key proof
 │   │   ├── mdi.py                # MDI-QKD proof
 │   │   ├── ma2005.py             # Ma et al. (2005) decoy-state
 │   │   ├── wang2005.py           # Wang (2005) proof
 │   │   ├── paper_2009_individual.py  # Individual-attack proof
 │   │   ├── optimization.py       # SLSQP pulse-intensity optimization
 │   │   └── utils_lp.py           # Linear programming solver helpers
 │   ├── utils/                    # Utility sub-package
 │   │   ├── math.py               # binary_entropy, confidence intervals
 │   │   ├── scm_analysis.py       # SCM/WDM crosstalk analysis
 │   │   ├── utils.py              # Sanitization for serialization
 │   │   └── validation.py         # Boolean parsing
 │   └── tests/                    # Package-level tests
 │       ├── test_physics_validation.py
 │       ├── test_qkd_network_pipeline.py
 │       └── test_sources.py
 ├── main_optimized.py             # Main simulation runner & sweep engine
 ├── main_optimized_dispatch.py    # Multiprocessing dispatch loop
 ├── qkd_network_7nodes.py         # 7-node trusted-relay QKD network
 ├── qkd_ui.py                     # Interactive UI runner
 ├── run_bridge.py                 # Entry-point bridge
 ├── validate_physics.py           # Physics validation harness
 ├── absolute_crypto_validation.py # Cryptographic validation
 ├── lp_validation.py              # LP solver validation
 ├── pulse_tracking_trace.py       # Pulse-level tracing
 ├── tests/                        # Top-level integration tests
 ├── scripts/                      # Verification scripts
 ├── qkd_results/                  # Output directory for sweep CSVs
 ├── simulation_results/           # Output directory for JSON results
 └── Report/                       # LaTeX report artefacts
 ```
 
 ## Key Modules & Implementation
 
 ### 1. Photon Sources (`qkd/sources.py`)
 
 - **`PoissonSource`** — standard attenuated laser with Poisson photon-number
   statistics.  Supports signal and decoy intensity configurations via
   `IntensityConfig` / `IntensityNode` dataclasses.
 - **`DensityMatrixSource`** — diagonal density-matrix sampling in the Fock
   basis.  Validates Hermiticity, positive semi-definiteness, and trace.
 - Classical intensity imperfections: MZM modulation (`MZMConfig`),
   electrical driver noise (`ElectricalNoiseConfig`), Gaussian intensity
   jitter, and adversarial-block source errors (`SourceErrorModel`).
 - Photon-number distributions: `poisson_pn_array`, `thermal_pn_array`,
   truncated Poisson PMF tail handling.
 
 ### 2. Optical Channel (`qkd/channel.py`)
 
 - **`FiberChannel`** — pure scalar Beer-Lambert attenuation:
   $T = 10^{-\alpha L / 10}$.  Optional wavelength consistency checks.
 - Explicitly documents limitations: no chromatic dispersion, no PMD/PDL,
   no nonlinear effects, no freespace links.  A `cascade` method supports
   multi-segment chains.
 - Strict mode (`contextvars.ContextVar`) rejects physically invalid
   configurations (zero transmittance, subnormal values).
 
 ### 3. Single-Photon Detectors (`qkd/detectors.py`)
 
 - **Event-driven threshold model** with dead time (non-paralyzable),
   afterpulsing (exponential decay), dark counts (Poisson process),
   timing jitter (Gaussian FWHM), and chromatic-dispersion broadening.
 - Three detector types: `SPD` (SPAD Geiger-mode), `SNSPD` (Planck-spectrum
   black-body dark-rate scaling), `PNRD` (photon-number-resolving with
   binomial thinning).
 - Gated-detector mode, asymmetric efficiencies, entangling decoder
   (CPTP-style visibility-aware dephasing), and Rogers _et al._ (2007)
   double-click policy.
 - Stateful batch simulation with serializable `DetectorRuntimeState`.
 
 ### 4. Protocols & Sifting (`qkd/protocols.py`)
 
 - **`BB84DecoyProtocol`** — standard decoy-state BB84 with Z/X basis
   selection, signal/decoy/vacuum intensities, and parameter estimation
   sub-sampling.
 - **`MDIQKDProtocol`** — Bell-state measurement model with coincident-click
   logic, anti-correlated Z-basis and correlated X-basis error rules.
 - **`B92Protocol`** and **`RedundantTransmissionProtocol`** for legacy
   and non-standard protocol variants.
 - Deterministic `SeedSequence`-based RNG derivation for reproducible
   parallel workers.
 
 ### 5. Security Proofs (`qkd/proofs/`)
 
 | Proof Module | Reference | Key Features |
 |---|---|---|
 | `lim2014.py` | Lim _et al._ (2014), PRA 89, 022307 | Paper-faithful 3-intensity; hybrid Hoeffding/Chernoff bounds; N-decoy LP fallback |
 | `tight.py` | BB84 tight finite-key | Tight entropy bounds; epsilon-smoothing |
 | `mdi.py` | MDI-QKD | Bell-state measurement analysis |
 | `ma2005.py` | Ma _et al._ (2005) | Vacuum+weak, 1-decoy, 2-decoy, asymptotic |
 | `wang2005.py` | Wang (2005) | Decoy-state with statistical fluctuation |
 | `paper_2009_individual.py` | Individual attacks | Legacy individual-attack security |
 | `optimization.py` | SLSQP | Joint optimisation of signal/decoy intensities and probabilities |
 
 The abstract base class `FiniteKeyProof` (`qkd/proofs/base.py`) provides:
 epsilon allocation, parameter estimation, binary entropy, safe
 divide/log/clamp, audit mode, and structured `KeyCalculationResult`.
 
 ### 6. Parameter Management (`qkd/params.py`)
 
 - **`QKDParams`** — a comprehensive dataclass validated at construction
   (`__post_init__`).  Fields span protocol, source, channel, detector,
   error correction, and security parameters.
 - Helper functions `load_lim2014_dedicated_params` and
   `load_lim2014_dwdm_params` re-create paper-faithful configurations
   with automatic distance-dependent launch-power computation.
 - Schema versioning (`CURRENT_SCHEMA_VERSION = "1.9"`), seed redaction,
   and recursive serialization depth limits.
 
 ### 7. Top-Level Runner (`main_optimized.py`)
 
 - `DEFAULT_CONFIG` dictionary with paper-faithful defaults (Lim2014
   Table I: 0.2 dB/km, 15% detector efficiency, 600 Hz dark rate,
   0.5% intrinsic QBER).
 - Builder functions (`build_source`, `build_detector`, `build_channel`,
   `build_protocol_parameters`) convert dict to typed objects.
 - `run_single_simulation` orchestrates: source -> channel -> protocol ->
   detection -> sifting -> security proof -> `SimulationResults`.
 - `run_sweep` and `main` provide CLI-driven parallel parameter sweeps via
   `multiprocessing.Pool` and `main_optimized_dispatch.dispatch_work_items`.
 
 ## Build & Installation
 
 ### Prerequisites
 
 - Python 3.10 or later
 - `pip` (Python package installer)
 
 ### Dependencies
 
 | Package | Minimum Version | Purpose |
 |---|---|---|
 | `numpy` | >=1.21 | Numerical arrays, RNG, linear algebra |
 | `scipy` | >=1.8 | Special functions, optimisation, statistics |
 | `mpmath` | (optional) | High-precision arithmetic for proof validation |
 
 ### Setup
 
 ```bash
 # Clone or unpack the repository
 cd "4.1 QKD Simulation Framework"
 
 # Install dependencies
 pip install numpy scipy
 
 # Optional: install mpmath for high-precision validation
 pip install mpmath
 ```
 
 No project-level `pyproject.toml` or `requirements.txt` is present;
 install the above packages manually.
 
 ### Quick Syntax Check
 
 ```bash
 python3 -m py_compile main_optimized.py qkd/*.py qkd/proofs/*.py
 ```
 
 ## Usage Guidelines
 
 ### Small Validation Run
 
 ```bash
 python3 main_optimized.py --workers 1 --min-pulses-log 4 --max-pulses-log 4
 ```
 
 Runs a single-threaded sweep with 10^4 pulses per configuration.
 
 ### Full Sweep
 
 ```bash
 python3 main_optimized.py \
     --workers 8 \
     --min-pulses-log 4 \
     --max-pulses-log 8 \
     --output-dir qkd_results
 ```
 
 ### Network Simulation
 
 ```bash
 python3 qkd_network_7nodes.py
 ```
 
 Simulates a 7-node QKD network with 10 direct links, 20 key pools,
 optical switches, SPF+ path selection, and XOR-based trusted-relay
 key delivery.  Loads `main_optimized.py` for per-link key generation.
 
 ### Running Tests
 
 ```bash
 python3 -m unittest discover qkd/tests
 python3 scripts/test_run_qkd.py
 python3 -m pytest tests/ -v
 ```
 
 ### Interactive UI
 
 ```bash
 python3 qkd_ui.py
 ```
 
 ## Output & Artifacts
 
 - **JSON results** — written atomically by `qkd/io.py` with optional
   gzip compression and SHA256 metadata.
 - **CSV sweep files** — one row per `(distance, num_pulses)` tuple.
 - **SimulationResults** — includes `params`, `metadata`,
   `security_certificate`, `decoy_estimates`, `secure_key_length`,
   `tally_stats`, and solver diagnostics.
 
 ## Scientific References
 
 - Lim, C. C. W., Curty, M., Walenta, N., Xu, F., and Zbinden, H.
   "Concise security bounds for practical decoy-state quantum key
   distribution." _Phys. Rev. A_ **89**, 022307 (2014).
 - Ma, X., Qi, B., Zhao, Y., and Lo, H.-K. "Practical decoy state for
   quantum key distribution." _Phys. Rev. A_ **72**, 012326 (2005).
 - Wang, X.-B. "Beating the photon-number-splitting attack in
   practical quantum cryptography." _Phys. Rev. Lett._ **94**, 230503
   (2005).
 - Rogers, D. J., Bienfang, J. C., Nakassis, A., Xu, H., and Clark,
   C. W. "Detector dead-time effects and paralyzability in
   high-speed quantum key distribution." _New J. Phys._ **9**, 319
   (2007).
 
 ## Codebase Statistics
 
 - **~50 Python source files** across the package, proofs, utilities,
   tests, and top-level scripts.
 - **~18,000 lines** in the three core physics modules (sources,
   detectors, channel).
 - **~4,500 lines** in the security proofs sub-package.
 - **~3,200 lines** in the main simulation runner.
 - **8 security proof implementations** spanning finite-key, asymptotic,
   and individual-attack regimes.
