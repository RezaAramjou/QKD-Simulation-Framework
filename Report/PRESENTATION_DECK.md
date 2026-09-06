# QKD Simulation Framework — 40-Minute Presentation Deck

**Presenter:** Reza Aramjou  
**Duration:** 40 minutes (18 slides × ~2 min + buffer)  
**Date:** August 2026

---

## Time Allocation

| Section | Slides | Time |
|---|---|---|
| Introduction & Motivation | 1–3 | 5 min |
| Architecture Overview | 4–5 | 5 min |
| Physical Layer | 6–8 | 7 min |
| Security Proofs | 9–11 | 7 min |
| Network Layer | 12–13 | 4 min |
| Validation & Testing | 14–15 | 4 min |
| Results & Conclusion | 16–18 | 8 min |

---

## SLIDE 1 — Title

**[Slide 1]: A Comprehensive QKD Simulation Framework — Bridging Quantum Theory and Software Engineering**

- **Visuals:** Full-bleed title. Framework emblem top-center. Alice → fiber → Bob diagram with lock icon. Author/date footer.
- **Slide Text:**
  - A Unified Simulation Framework for Discrete-Variable Quantum Key Distribution
  - From Optical Pulse Generation to Composable Security Proofs and Network Routing
  - Reza Aramjou — August 2026
- **Script:** "Good morning, everyone. Thank you for being here. Today I'm going to present a comprehensive simulation framework for Quantum Key Distribution — or QKD — that I have developed over the past several months. This is not merely a key-rate calculator. It is a full-stack research infrastructure that models the entire pipeline from the generation of individual optical pulses, through the physics of fiber-optic channels and detector imperfections, all the way up to composable finite-key security proofs, linear-programming-based parameter optimization, and multi-node trusted-relay network routing. By the end of this talk, I hope you will appreciate both the depth of physical modeling that this framework achieves and the engineering discipline — the layered architecture, the rigorous validation methodology, the numerical safeguards — that makes it reliable as a research tool. Let me start by explaining why a framework like this is needed."

---

## SLIDE 2 — Motivation

**[Slide 2]: Motivation — Why QKD Matters Now**

- **Visuals:** Left: timeline RSA→Shor→Harvest Now Decrypt Later. Right: QKD diagram Alice/Bob/Eve with no-cloning callout. Bottom: Computational vs Information-Theoretic security table.
- **Slide Text:**
  - Classical public-key cryptography relies on computational hardness (factoring, discrete log)
  - Shor's algorithm breaks RSA/ECC in polynomial time on a fault-tolerant quantum computer
  - "Harvest Now, Decrypt Later": data intercepted today can be decrypted tomorrow
  - QKD offers information-theoretic security based on the laws of quantum mechanics
  - Complementary to Post-Quantum Cryptography; QKD + PQC coexist in hybrid architectures
- **Script:** "Let me set the stage. The security of virtually all deployed public-key cryptography — RSA, elliptic-curve cryptography — rests on the computational difficulty of factoring large integers or solving discrete logarithms. These are problems that a sufficiently powerful fault-tolerant quantum computer, running Shor's algorithm, can solve in polynomial time. The threat is not merely theoretical. There is the well-known 'Harvest Now, Decrypt Later' scenario: an adversary can record encrypted traffic today and store it, waiting until quantum computers become available to retroactively decrypt that data. Quantum Key Distribution addresses this by shifting the security foundation from computational assumptions to the laws of physics. In QKD, any attempt by an eavesdropper to measure the quantum states en route necessarily disturbs those states in a detectable way. The result is information-theoretic security — security that holds regardless of the adversary's computational power. Importantly, QKD is not a replacement for all classical cryptography. It coexists with post-quantum cryptography; QKD handles the key distribution problem with physics, while PQC secures the classical authentication channel and other components."

---

## SLIDE 3 — The Gap

**[Slide 3]: Bridging the Theory–Engineering Divide**

- **Visuals:** Two columns: "Theoretical QKD Papers" (asymptotic formulas, ideal detectors, infinite pulses) vs "This Framework" (finite-key, dead-time cascades, afterpulsing, DWDM, LP optimization, network routing). Arrow: "~1800 points, 10 campaigns."
- **Slide Text:**
  - Most tools oversimplify physics or hard-code a single security proof
  - Real-world effects neglected: dead-time cascades, afterpulsing, DWDM cross-talk
  - Asymptotic analyses ignore finite-key statistics — essential for practical systems
  - This framework unifies: rigorous physics + multiple composable proofs + network orchestration
  - Design principles: layer separation, immutability, fail-fast validation, single source of truth
- **Script:** "There is a substantial gap between how QKD is described in theoretical papers and what is needed to build a trustworthy simulation. Most published protocols derive asymptotic key-rate formulas under idealized assumptions: perfect single-photon sources, detectors with no dead time, infinite pulses. Many existing simulation tools compound the problem by hard-coding a single security proof and using simplified detector models. The framework I built closes this gap systematically. It models the physical layer with genuine rigor: fiber attenuation follows Beer-Lambert with numerical safeguards against underflow, detectors have configurable dead-time models and afterpulsing, sources support decoy-state protocols with multiple intensity levels, and the channel layer includes DWDM nonlinear cross-talk. On top of this physical foundation, the framework supports four distinct composable security proofs — Lim 2014, Ma 2005, Wang 2005, and a Tight proof — all operating on a common FiniteKeyProof abstract interface. Finally, the framework extends to a 7-node trusted-relay network with SPF+ routing and key-pool management. In total, the framework has been exercised across ten distinct simulation campaigns encompassing roughly 1,800 individual simulation points."

---

## SLIDE 4 — Architecture

**[Slide 4]: System Architecture — A Layered, Modular Design**

- **Visuals:** Four-layer vertical stack: Network (SPF+, key pools), Execution (workers, sweeps, SLSQP), Cryptographic (FiniteKeyProof ABC, 4 proofs, LP), Physical (sources, detectors, channel, noise). Data flow arrows between layers.
- **Slide Text:**
  - Physical Layer: Coherent/thermal sources, SNSPD/SPD/PNRD detectors, FiberChannel with Beer-Lambert, noise models
  - Cryptographic Layer: FiniteKeyProof ABC, four proofs, epsilon-budget allocation, LP parameter optimization
  - Execution Layer: Parallel workers, log/linear sweeps, auto-optimization via SLSQP, CSV output
  - Network Layer: 7-node trusted-relay topology, 10 links, 20 key pools, SPF+ routing, failover
- **Script:** "Let me walk you through the four-layer architecture. At the bottom is the Physical Layer, implemented in the qkd Python package — optical source models, detector models including SPD, SNSPD, and PNRD with dead-time and afterpulsing, and the fiber channel with Beer-Lambert attenuation and noise models. Above that sits the Cryptographic Layer, centered on the FiniteKeyProof abstract base class. Every security proof inherits from this base and must implement four abstract methods. Concrete implementations include Lim 2014, Ma 2005, Wang 2005, and a Tight proof. There is also a linear programming solver that optimizes protocol parameters to maximize key rate. The Execution Layer, embodied in main_optimized.py, orchestrates parallel workers, logarithmic and linear sweeps, and automatic SLSQP optimization. Finally, the Network Layer implements a 7-node QKD network using trusted relays, with 10 fiber links, 20 independent key pools, and SPF+ routing. Each layer communicates through well-defined frozen dataclasses — QKDParams, TallyCounts, DecoyEstimates, KeyCalculationResult — ensuring layers remain decoupled and independently testable."

---

## SLIDE 5 — Configuration

**[Slide 5]: QKDParams — Centralized, Immutable, Validated Configuration**

- **Visuals:** QKDParams frozen dataclass diagram with nested sub-configs. Validation flow: User Dict → from_dict() → _validate() → Frozen QKDParams. Schema version "1.9" badge. with_channel_rebuilt() example.
- **Slide Text:**
  - QKDParams is the single entry point (~1,700 lines, type-driven)
  - Deep validation of 50+ parameters at construction (fail-fast)
  - Immutable: @dataclass(frozen=True) — no accidental mutation
  - Schema versioning (v1.9) ensures cross-version reproducibility
  - Factory presets reproduce published Lim 2014 benchmarks
  - Serialization: to_serializable_dict() ↔ from_dict()
- **Script:** "A key architectural decision is the centralization of all configuration into a single, immutable, deeply validated data structure called QKDParams — a frozen Python dataclass over 1,700 lines. When constructed, the _validate() method runs over fifty checks. If any fails, construction raises an explicit exception immediately — fail-fast. The object is frozen after construction. If you need a modified configuration, you use with_channel_rebuilt(), which returns a new validated instance. The framework supports serialization round-trips and includes schema versioning at version 1.9. Pre-built factory presets reproduce published benchmark configurations from the Lim 2014 paper, including dedicated-fiber and DWDM scenarios, ensuring reproducibility."

---

## SLIDE 6 — Channel Physics

**[Slide 6]: FiberChannel — Immutable Physics with Numerical Safeguards**

- **Visuals:** Equation T = 10^(−αL/10). Graph: log transmittance vs distance with underflow region annotation. Sidebar: numerical protections list. Noise model hierarchy.
- **Slide Text:**
  - Core: T = 10^(−αL/10), α ≈ 0.2 dB/km at 1550 nm
  - All channel objects immutable — frozen dataclass
  - Numerical safeguards: subnormal clamping, safe_log, underflow detection
  - Strict mode: warnings → hard exceptions for production
  - ElectricalNoiseConfig: Johnson-Nyquist + Schottky shot noise
  - DWDM nonlinear cross-talk for multi-channel
- **Script:** "The fundamental channel model is Beer-Lambert: transmittance T equals ten to the power of negative alpha-L over ten, with alpha approximately 0.2 dB/km at 1550 nm. For long distances beyond 300-400 km, transmittance becomes astronomically small — below IEEE-754 subnormal thresholds — and can underflow to zero, silently corrupting entropy calculations. The framework detects these conditions and applies safe clamping. The FiberChannel class is a frozen dataclass — once constructed, its properties are permanently fixed. A separate ElectricalNoiseConfig dataclass models Johnson-Nyquist thermal noise and Schottky shot noise. For DWDM, nonlinear cross-talk is modeled. A strict mode flag upgrades physical-suspicion warnings into hard exceptions for automated production pipelines."

---

## SLIDE 7 — Sources & Detectors

**[Slide 7]: Optical Sources and Detector Models**

- **Visuals:** Left: CoherentSource (Poisson), ThermalSource, decoy-state levels. Right: Detector comparison table — SPD, SNSPD, PNRD with efficiency, dark count, dead time, afterpulse. Highlight SNSPD.
- **Slide Text:**
  - Coherent Source: Poisson photon statistics, configurable μ per pulse
  - Decoy-state: signal (μ ≈ 0.4–0.8), decoy (ν ≈ 0.1–0.2), vacuum
  - SPD: η ~0.15–0.3, Y₀ ~10⁻⁶/gate, moderate cost
  - SNSPD: η ~0.5–0.9, Y₀ ~10⁻⁸/gate, cryogenic, best long-distance
  - PNRD: photon-number-resolving for advanced protocols
  - All detectors: configurable dead-time models and afterpulsing
- **Script:** "The framework models coherent laser sources with Poisson-distributed photon numbers. For the decoy-state protocol — critical for defeating photon-number-splitting attacks — the source emits at signal, decoy, and vacuum intensities. On the detector side: SPDs with efficiencies around 15-30% and dark counts around 10⁻⁶ per gate; SNSPDs with 50-90% efficiency and dark counts as low as 10⁻⁸ per gate but requiring cryogenic cooling; and PNRDs for protocols needing photon-number resolution. Crucially, the framework models dead-time behavior realistically — when pulse rates are high, dead-time effects cascade, creating a measurable nonlinear reduction in effective detection efficiency. Afterpulsing is also modeled. These imperfections directly affect QBER and, through the security proof, the final secure key rate."

---

## SLIDE 8 — Protocol Pipeline

**[Slide 8]: Protocol Pipeline — From Pulses to Sifted Keys**

- **Visuals:** Flow: Source → Channel → Detection → Basis Sifting → Error Estimation → Key Distillation. Basis diagram: Z-basis {|0⟩,|1⟩}, X-basis {|+⟩,|−⟩}. Tally-key categories table.
- **Slide Text:**
  - BB84: Alice prepares Z/X basis; Bob measures randomly
  - Basis sifting discards ~50% of detections via classical channel
  - Decoy-state: estimates Y₁ (single-photon yield) and e₁ (error) from intensity statistics
  - Tally counts by (basis, intensity) pair feed security proofs
  - ProtocolParameters dataclass provides uniform abstraction
- **Script:** "In the BB84 protocol with decoy states, Alice randomly selects a basis and intensity level each round. Bob independently chooses a measurement basis. After quantum transmission, basis sifting discards mismatched-basis rounds — roughly half. The remaining events are sorted into tally categories by basis and intensity. These tally counts drive the decoy-state analysis: because Eve cannot distinguish which intensity was chosen, comparing detection rates across intensity levels lets us bound the single-photon yield Y₁ and error e₁. These estimates feed into the finite-key security analysis. All protocol parameters are encapsulated in a ProtocolParameters dataclass that plugs uniformly into the simulation pipeline regardless of which protocol or proof is used."

---

## SLIDE 9 — FiniteKeyProof ABC

**[Slide 9]: FiniteKeyProof — The Abstract Security Framework**

- **Visuals:** UML: FiniteKeyProof ABC with abstract methods: notation_map(), allocate_epsilons(), get_epsilon_policy(), estimate_yields_and_errors(). Template Method: compute_key_rate(). Data contracts: TallyCounts → DecoyEstimates → KeyCalculationResult. Epsilon decomposition tree.
- **Slide Text:**
  - All proofs inherit from FiniteKeyProof ABC — uniform interface, comparable outputs
  - Template Method: compute_key_rate() orchestrates ε-allocation → yield estimation → key-length
  - Epsilon decomposition: ε_sec divided among error correction, privacy amplification, parameter estimation
  - TallyCounts: raw (basis, intensity) counts
  - DecoyEstimates: Y₁^L, e₁^U, feasibility flag
  - KeyCalculationResult: final secure key length, leak breakdown, diagnostics
- **Script:** "All security proofs inherit from a single abstract base class called FiniteKeyProof. This enforces a uniform interface — every proof exposes the same methods and produces the same output data structures, which is essential for fair cross-proof comparisons. The base follows the Template Method pattern. compute_key_rate orchestrates: allocate_epsilons decomposes the total security parameter into sub-budgets; estimate_yields_and_errors computes statistical bounds on single-photon yield and error using configurable confidence methods; calculate_key_length computes the secure key length accounting for error-correction leakage and privacy amplification. Data flows through three frozen dataclasses: TallyCounts holds raw counts, DecoyEstimates holds inferred single-photon parameters, and KeyCalculationResult provides the final key length with leak breakdown and error codes. Each stage can be tested independently — critical for cryptographic software where correctness must be demonstrable."

---

## SLIDE 10 — Four Proofs

**[Slide 10]: Proof Comparison — Lim 2014, Ma/Wang, and Tight Proof**

- **Visuals:** Four-quadrant layout. Each: proof name, simplified key formula, distinguishing features. Center: "Same Physical Engine + Same TallyCounts → Fair Comparison."
- **Slide Text:**
  - Lim 2014: Dedicated + DWDM presets, optimized for realistic fiber
  - Ma 2005: Foundational decoy-state proof, rigorous Y₁/e₁ framework
  - Wang 2005: Alternative statistical bounding, different variance treatment
  - Tight Proof: Asymptotically optimal rates, minimal ε-leakage overhead
  - All proofs use the same physical simulation engine
  - Differences attributable solely to cryptographic assumptions
- **Script:** "The framework implements four concrete security proofs. The Lim 2014 proof handles both dedicated-fiber and DWDM scenarios with finite-key corrections and factory presets. The Ma 2005 proof provides the foundational decoy-state security analysis. The Wang 2005 proof offers an alternative statistical treatment. The Tight proof achieves rates approaching the asymptotic optimum by minimizing epsilon-leakage overhead. The critical point: all four operate on the same physical simulation engine. When I compare key rates from Lim 2014 versus Ma 2005 for identical configurations, any difference is attributable solely to cryptographic assumptions — not to different physical models. This controlled comparison is extremely difficult with separate codebases."

---

## SLIDE 11 — LP Optimization

**[Slide 11]: Linear Programming — Optimizing Protocol Parameters**

- **Visuals:** LP formulation: decision variables (μ, ν, probabilities), constraints matrix, objective function. SLSQP flow diagram. Before/after optimization key rate comparison plot.
- **Slide Text:**
  - LP formulation: maximize key rate subject to decoy-state + physical constraints
  - SLSQP solver (scipy.optimize.minimize) for non-linear extensions
  - Campaign 4: auto-optimization finds optimal μ, ν, probabilities per distance
  - Constraints: Y₁ ≥ 0, e₁ ≤ 0.5, μ > ν > 0, Σp = 1.0
  - Result: ~15–25% key rate improvement over manual parameters at intermediate distances
- **Script:** "One of the most powerful features is integrated parameter optimization. Given a fixed fiber length and detector, what combination of intensities maximizes the secure key rate? We formulate this as constrained optimization with SLSQP. The decision variables are source intensities and their probabilities. Constraints come from decoy-state analysis and physical limits. Campaign 4 is dedicated to auto-optimization: for each distance, multiple SLSQP iterations evaluate the full pipeline to converge to optimal parameters. Results show 15 to 25 percent higher key rates compared to manually chosen intensities, especially at intermediate distances where the signal-versus-yield trade-off is most sensitive."

---

## SLIDE 12 — Network

**[Slide 12]: Network Layer — Trusted-Relay QKD at Metropolitan Scale**

- **Visuals:** 7-node topology: A (receiver), B/C (relays), D/E/F/G (senders). 10 links with distances. Key pool icons at each node. Legend: active/hot-standby.
- **Slide Text:**
  - 7 nodes: 4 senders, 2 trusted relays, 1 receiver
  - 10 fiber links, 20 independent key pools (2 per link)
  - Trusted Relay: key hops through intermediate nodes; relays must be physically secured
  - Two backends: full pipeline (PipelineQKDLinkSimulator) + analytical fallback
  - Key pools: LOW watermark triggers replenishment, HIGH stops
  - Periodic background replenishment keeps all pools healthy
- **Script:** "Scaling up to a full network: a 7-node metropolitan-scale trusted-relay architecture. Four senders — D through G — connect through two relays — B and C — to receiver A, with ten fiber links and twenty key pools. The trusted relay model is the standard practical approach for extending QKD range. Each relay decrypts and re-encrypts key material, so relays must be physically secured. Each key pool has configurable watermarks — LOW triggers replenishment, HIGH pauses it. A background task continuously monitors pools. The framework supports two backends: the full physical pipeline for accuracy, and a lightweight analytical fallback for rapid prototyping."

---

## SLIDE 13 — SPF+ Routing

**[Slide 13]: SPF+ — Shortest-Path-First with Key-Pool Awareness**

- **Visuals:** Network with dynamic edge weights: normal=1, depleted=∞. Walkthrough: D→A primary D-B-A (cost 2); if B-A depleted → D-C-A. Pseudocode box. Failover demo: B-D DOWN → reroute via C.
- **Slide Text:**
  - SPF+ extends Dijkstra with dynamic link costs based on key-pool fill ratio
  - Link cost = base metric + penalty ∝ 1/(fill ratio)
  - Depleted pools → cost = ∞ → link excluded
  - Automatic failover: DOWN link → immediate SPF+ recalculation
  - Complexity: O((V+E) log V) ≈ 50 ops per request — negligible
  - Demo: B-D failure → D-A reroutes via C, then recovery
- **Script:** "SPF+ extends Dijkstra with dynamic costs based on key-pool availability. When a pool is full, cost equals base metric. As it drains, a penalty inversely proportional to fill ratio increases cost. Below CRITICAL watermark, cost becomes infinity, removing the link. This provides automatic load balancing and failover. In the demo, link B-D fails — D-to-A requests automatically reroute through C. After restoration and replenishment, traffic returns to the shorter path. With seven nodes, each SPF+ invocation requires about 50 operations — negligible overhead, suitable for real-time routing."

---

## SLIDE 14 — Validation: Physics & Crypto

**[Slide 14]: Validation Strategy — Physics Benchmarks and Cryptographic Checks**

- **Visuals:** Left: detection probability vs theoretical, invariants table. Right: Rogers et al. benchmark overlay (Campaign 9), epsilon-consistency check diagram.
- **Slide Text:**
  - Physical invariants: monotonicity, round-trip symmetry, noise additivity
  - Beer-Lambert validated 0–500 km, relative error < 10⁻¹²
  - Rogers et al. benchmark: reproduced published key-rate curves (Campaign 9)
  - Epsilon self-consistency: Σ ε_i ≤ ε_sec for all proofs
  - Y₁/e₁ estimates verified against analytical asymptotic bounds
  - 50+ parameter checks in QKDParams._validate()
- **Script:** "Validation is built into every layer. Physical invariants: transmittance must decrease monotonically, round-trip must recover original transmittance, noise variances must add linearly. Beer-Lambert implementation validated against closed form across 0–500 km with relative errors below 10⁻¹². Cryptographic validation: Campaign 9 reproduces the Rogers et al. reference curves for decoy-state BB84 — our simulated key rates overlay the published data. Epsilon-budget self-consistency is verified: for every proof, allocated sub-epsilons sum to at most epsilon_sec. Single-photon estimates are verified against analytical asymptotic bounds. And fifty-plus parameter checks run at every QKDParams construction."

---

## SLIDE 15 — Software Validation

**[Slide 15]: Software Validation — Concurrency and LP Solver Integrity**

- **Visuals:** Left: concurrency test results — worker isolation, deterministic output, key-pool atomicity. Right: LP solver test cases table — simple, degenerate, decoy-state, failure injection.
- **Slide Text:**
  - Worker isolation: sub-counts sum to single-worker total
  - Deterministic reproducibility: fixed seed → bit-identical CSV with 1/2/4/8 workers
  - Key-pool atomicity: concurrent ops maintain total key count invariant
  - LP solver: 20+ test cases — simple, degenerate, decoy-state
  - Failure injection: infeasible, unbounded, ill-conditioned → graceful diagnostics
  - SolverDiagnostics dataclass captures status, iterations, violations
- **Script:** "Beyond physics and cryptography, the framework undergoes rigorous software validation. Concurrency tests verify worker isolation, deterministic reproducibility with fixed seeds regardless of worker count, and key-pool atomicity under concurrent access. The LP solver is validated against over twenty test cases — simple problems with known optima, decoy-state estimation problems, degenerate cases, and failure scenarios including infeasible and unbounded LPs. Each test verifies correct solver status and constraint satisfaction. SolverDiagnostics captures status, iterations, and violations for post-hoc auditing of every LP solve."

---

## SLIDE 16 — Campaigns Overview

**[Slide 16]: Ten Campaigns, ~1,800 Simulation Points**

- **Visuals:** Summary table: Campaign #, Name, Sweep Variable, # Points. Color-coded by category: baseline, scaling, DWDM, comparisons, validation, sensitivity. Note: each point = full physical + proof run.
- **Slide Text:**
  - C1: BB84+Lim2014 baseline — 96 pts (distance sweep)
  - C2: Finite-key scaling — 288 pts (pulses 10⁴→10⁸ × distance)
  - C3: DWDM multi-channel — 288 pts (power −20/−34/−40 dBm)
  - C4: Auto-optimized (SLSQP) — 96 pts
  - C5: 4-protocol comparison — 384 pts
  - C6: 4-proof comparison — 384 pts
  - C7: 3-detector comparison — 288 pts
  - C8: Confidence-bound comparison — 288 pts
  - C9: Rogers et al. validation — 96 pts
  - C10: Sensitivity (QBER × misalignment) grid
- **Script:** "The framework was exercised through ten distinct simulation campaigns totaling approximately 1,800 points. Each point is a complete run: pulse generation, channel transmission, realistic detection, basis sifting, statistical parameter estimation, decoy-state LP, and finite-key security proof. Campaigns 1-4 cover baselines, scaling, DWDM, and optimization. Campaigns 5-8 perform controlled comparisons across protocols, proofs, detectors, and confidence methods. Campaign 9 validates against published benchmarks. Campaign 10 performs two-dimensional sensitivity analysis over QBER and misalignment. Each campaign produces a single CSV file as the authoritative record, ensuring complete reproducibility."

---

## SLIDE 17 — Key Findings

**[Slide 17]: Key Findings — What the Simulations Tell Us**

- **Visuals:** 2×2 grid: Detector comparison (Fig 7), Protocol comparison (Fig 5), Proof comparison (Fig 6), Sensitivity heatmaps (Fig 10a/10b). Annotated takeaways.
- **Slide Text:**
  - Detectors: SNSPD doubles secure range vs SPD; PNRD for advanced protocols
  - Protocols: Decoy-state BB84 outperforms non-decoy variants
  - Proofs: Tight proof yields ~10–15% higher rate than Lim 2014; Ma/Wang intermediate
  - Confidence: Clopper-Pearson most conservative; Gaussian maximizes rate for large N
  - Sensitivity: 0.01 rad misalignment → 50% key rate reduction
  - Optimization: SLSQP-optimized intensities → 15–25% improvement at intermediate distances
- **Script:** "Key findings. Detectors: SNSPDs double the secure distance versus SPDs, primarily due to thousand-fold lower dark counts. Proofs: the Tight proof achieves 10-15% higher key rates than Lim 2014 at equivalent security — tighter statistical bounds directly translate to performance. Confidence methods: Clopper-Pearson is most conservative; Gaussian maximizes rates for large N. For N below 10⁵, Clopper-Pearson is recommended; beyond 10⁶, Gaussian is well justified. Sensitivity: a misalignment of just 0.01 radians — about half a degree — can reduce key rate by 50%, emphasizing the critical importance of optical alignment. SLSQP optimization yields 15-25% higher key rates compared to manual parameters, particularly at intermediate distances."

---

## SLIDE 18 — Conclusion

**[Slide 18]: Summary and Outlook**

- **Visuals:** Three columns: "Achievements" (checkmarks), "Key Takeaways" (3-4 bullets), "Future Work" (MDI-QKD, repeaters, KMS, ML, CV-QKD). Closing: "Questions?" with contact.
- **Slide Text:**
  - Framework: Complete DV-QKD infrastructure — 4 layers, 4 proofs, 3 detector types, 7-node network
  - Engineering: Immutable config, fail-fast validation, composable contracts, deterministic reproducibility
  - Validation: Physics benchmarks, crypto cross-checks, Rogers reproduction, concurrency correctness
  - Key Insight: SNSPD + Tight proof + optimized intensities = best practical performance
  - Future: MDI-QKD, quantum repeaters, real-time KMS, ML optimization, CV-QKD extension
- **Script:** "Let me conclude. I have presented a comprehensive DV-QKD simulation framework spanning four architectural layers, four composable security proofs, three detector types with realistic imperfections, and a 7-node trusted-relay network with key-pool-aware routing. The framework is engineered with production scientific software rigor. The key practical insight: combine SNSPD detectors with the Tight security proof and SLSQP-optimized intensities for maximum rate and range — doubling secure distance and improving key rate by 15 to 25 percent. Future extensions include MDI-QKD, quantum repeaters, real-time KMS integration, machine-learning parameter optimization, and a CV-QKD module. This framework is, at its core, a bridge between the theoretical formulations of QKD security and the practical, engineered reality of quantum communication systems. Thank you. I am happy to take your questions."
