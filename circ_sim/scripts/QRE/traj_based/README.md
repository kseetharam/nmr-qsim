# Trajectory-based quantum resource estimation: goal

## Context

`../pipeline_plan.md` builds a (fault-tolerant, platform-agnostic) resource
estimate for **explicit Lindbladian simulation**: the dissipative jump
operators $L_j$ are dilated into an ancilla register and block-Trotterized
alongside the coherent Hamiltonian $H_0$ (`coherent_trotter_step.py` +
`nested_trotter_step.py`, dispatched via unary iteration,
`unary_iteration.py`). That pipeline now has a first complete baseline
(2026-09-22): $\approx1.36\times10^9$ T-gates, 14 peak qubits (5 system + 5
dispatch-select + 4 flag ancilla) for one circuit shot of truncated
Gemcitabine's ZULF protocol.

The **trajectory-based approach** is an alternative encoding of the same
physics, motivated by `../../trajectory_based_sim/notes/white_noise_trotter_1.pdf`:
in the white-noise limit relevant here (rotational correlation time
$\tau_c\sim$ ns $\ll\Delta t$), the Lindbladian dissipator $\mathcal D$ can be
generated *exactly*, after classical disorder-averaging, by drawing Gaussian
Wiener increments $\Delta W_j\sim\mathcal N(0,\gamma_j\Delta t)$ per Trotter
step and propagating under the purely coherent, but now *random*, generator

$$U(\Delta t) = \exp\!\Big[-i\big(H_0\Delta t + \textstyle\sum_j \Delta W_j V_j\big)\Big]$$

(or a Lie/Strang splitting of $H_0$ and $\sum_j\Delta W_j V_j$), where $V_j$
are Hermitian noise generators built from the canonical (generally
non-Hermitian) jump operators (`../../trajectory_based_sim/testing/trajectory_convergence.py`'s
`build_hermitian_generators`, verified exact for Gemcitabine's full
25-operator dissipator to machine precision). Averaging the resulting
per-trajectory circuit output over many classical noise draws reproduces
$\langle O\rangle$ under the true Lindbladian, with no dilation ancilla and
no jump-operator dispatch register at all -- the dissipative physics is
pushed entirely into classical randomness over an otherwise-coherent
circuit.

**This is deliberately not "sample a molecular orientation trajectory."**
`../../trajectory_based_sim/notes/route1_rotdiff.pdf` ("Route 1") is the
literal alternative -- draw $R(t)\in SO(3)$ as Brownian motion and hold it
piecewise-constant per step, so the instantaneous Hamiltonian is
$H(R_k)=\sum_{ab}\mathcal D_{ab}(R_k)W_{ab}$ with $\mathcal D(R)$ a genuine
rotation matrix mixing which dipolar/CSA lab-frame components are large at
that instant. That note explicitly rules Route 1 out for liquid-state NMR on
cost grounds alone (Sec. 6: direct simulation would need $\sim5\times10^{10}$
steps/trajectory at realistic $\tau_c$) and instead takes the $\tau_c\to0$
white-noise limit, where "the orientation drops out of the problem" (Sec. 4)
-- replaced by the *fixed* generator set $\{V_j\}$ above, each entering with
only an i.i.d. Gaussian **scalar** $\Delta W_j$, not a rotation. This
pipeline builds on that white-noise/fixed-generator picture (matching
`trajectory_convergence.py`), not on literal orientation sampling.

This has already been derived analytically (Trotter/splitting error for all
three schemes, `white_noise_trotter_1.pdf` Secs. 2-5: $O(\gamma T\Delta t)$
Lie, $O(T\Delta t^2)$ Strang, a coherent $O(\gamma\Delta t)$ frequency-shift
for the unsplit scheme, single-trajectory strong-order-one convergence
$\propto\sqrt\gamma$ for all three) and checked numerically both on a
single-qubit toy model (same note, Sec. 6) and on the real truncated-Gemcitabine
ZULF system (`../../trajectory_based_sim/testing/{trajectory_convergence,plot1_trajectory_convergence,plot2_trajectory_vs_dt}.py`,
using the unsplit scheme, `notes/route1_rotdiff.pdf` for the physical
white-noise-limit justification). What doesn't exist yet is a resource
estimate: how many T-gates, qubits, and — the genuinely new ingredient this
approach introduces — classical noise **trajectories**, are needed to reach
a target precision on the same observable ($\langle\text{coil}\rangle$,
baseline: gyromagnetic-weighted collective magnetization) the Lindbladian
pipeline already targets.

## Goal

Set up a resource-estimation pipeline for the trajectory-based approach,
structured the same way as `../pipeline_plan.md` (input spec -> algorithmic
primitive -> error budget composition -> resource counting, deliberately
platform-agnostic for now), so the two approaches produce **directly
comparable** end-to-end resource totals for the same molecule/protocol/
target precision. Reuse rather than duplicate what already exists and
applies unchanged:

- **Physical inputs.** `../molecule_operators.py`'s `load_molecule_operators`
  (edge-colored $H_0$, canonical jump operators, $\rho_0$, coil) — the
  Hermitian-generator construction is a classical-data transform on top of
  the same jump operators, not a new physical model.
- **Coherent-step gate compilation.** `../coherent_trotter_step.py`'s
  Pauli-exponential/CNOT-ladder/rotation machinery and
  `../rotation_synthesis_strategy.py`'s `RotationSynthesisStrategy`
  abstraction both apply directly to the combined generator's Pauli terms —
  there is no new gate-synthesis primitive here, only a different set of
  terms (and randomly-drawn, rather than fixed, rotation angles) feeding the
  same compilation gadgets.
- **Readout.** `../coil_readout.py`'s direct X-/Y-basis measurement of
  $S_x^{\text{coll}}$/$S_y^{\text{coll}}$ is unchanged; the trajectory
  approach only changes what happens *before* readout.

## What's actually new here

1. **No dilation ancilla, no dispatch register.** The single largest
   structural difference from `../pipeline_plan.md`'s pipeline: dissipation
   is classical randomness applied to a coherent circuit, not a unitary
   dilation of jump operators. If this holds up under a real resource count,
   the qubit count should drop to just the system register (vs. 14 for
   Gemcitabine in the Lindbladian pipeline) at the cost of the sampling
   overhead in point 3.
   **Structurally confirmed, 2026-09-29** (pending a direct `QubitCount`
   check on an actual Qualtran circuit, not yet built): every numerical
   gadget built so far (`../../trajectory_based_sim/testing/plot3_lie_eq8_verification.py`
   through `plot7_group_trotter_fid_convergence.py`) operates purely on the
   $D_\text{sys}=32$ (5-qubit) system Hilbert space; no ancilla of any kind
   appears anywhere in the construction.
2. **Splitting-scheme choice is now a genuine resource-relevant decision.**
   Unlike the Lindbladian pipeline (whose nested-Trotter mitigation question
   was ZULF-specific), `white_noise_trotter_1.pdf` Secs. 2-3 give closed-form
   leading-order error coefficients for Lie ($O(\gamma T\Delta t)$, cancels
   under order-alternation), Strang ($O(T\Delta t^2)$, no extra gate cost
   over Lie), and unsplit ($O(\gamma\Delta t)$ but purely coherent, i.e.
   absorbable into a redefined $H_0$) — each with different per-step circuit
   structure (one exponential for unsplit vs. two half-step $H_0$
   exponentials bracketing the noise exponential for Strang). The
   error-budget layer needs to actually make this trade-off, not just fix
   unsplit because that's what the exploratory numerics used.
   **Baseline chosen, 2026-09-29 (see Decision log): first-order Lie, with a
   required `TrotterScheme` abstraction so Strang is a swap, not a rewrite.**
   Not derived from an error budget (explicitly deferred, see Design
   constraints) — chosen so the Qualtran machinery has something concrete to
   be built against now.
3. **A second, genuinely new sampling axis: trajectory count $N$.** The
   existing pipeline already deferred "how many measurement shots $M$ to
   converge $\langle O\rangle$" (`../pipeline_plan.md`'s decision log). This
   approach adds a *second*, distinct sampling question on top: how many
   independent classical noise trajectories are needed so the
   trajectory-averaged estimate converges to the disorder-averaged
   $\langle O\rangle$ within budget, given `white_noise_trotter_1.pdf`
   Sec. 4's explicit trade-off between the deterministic splitting bias
   $\Delta O$ and the statistical term $\sigma_O/\sqrt N$ (need
   $N\gtrsim(\sigma_O/\Delta O)^2$ to resolve the bias at all) — and how $N$
   composes with $M$ (fresh circuit realization per trajectory, but
   potentially many shots per realization). Resolving this is this
   sub-pipeline's analogue of open question/decision-log item on shot
   budgets in `../pipeline_plan.md`, not a re-opening of it.
   **Empirically explored, not yet solved, 2026-09-29**: see Numerical
   validation to date below for concrete $(N,\varepsilon)$ data points this
   trade-off will need to reproduce/generalize.
4. **Per-circuit resource count is plausibly, but not yet verifiably,
   trajectory-independent — and the argument is narrower than it first
   looks.** Each trajectory only differs in the drawn scalars $\Delta W_j$
   multiplying an already-*fixed* generator $V_j$ (fixed Pauli-term
   decomposition, molecule-derived, identical across trajectories — this is
   what distinguishes the white-noise picture above from literal
   orientation sampling, where the *pattern* of which couplings dominate
   genuinely changes every step). A sign flip of $\Delta W_j$ shifts every
   one of $V_j$'s internal term phases by exactly $\pi$, which is already
   inside the phase-mod-$\pi$ equivalence class `../nested_trotter_grouping.py`
   groups by — so compatible-term fusion *within* one generator's own Pauli
   terms is invariant under the randomness, and only rotation *angles* (not
   gate counts, at fixed $\varepsilon_{\text{gate}}$) vary trajectory to
   trajectory. This argument does **not** cover fusing terms *across*
   different generators $V_j,V_k$ — that would depend on the relative
   sign/phase of two independent random draws and would genuinely be
   trajectory-dependent. Whether this pipeline's compilation ever attempts
   such cross-generator fusion is an open design choice, not yet ruled in or
   out; until it's settled, the "one representative circuit's cost $\times$
   $M$ $\times$ $N$ $\times$ shots" factoring should be treated as a
   hypothesis to check (the same way `../pipeline_plan.md`'s $M$-step
   linearity was checked, not assumed), specifically including a check that
   no cross-generator grouping is silently relied on before trusting it.
   **Resolved 2026-09-29, for the real-coefficient case this pipeline
   actually uses.** Cross-generator fusion turns out to be exactly safe
   here, for a reason specific to this (ancilla-free, real-coefficient)
   setting: commutation between two Pauli strings never depends on their
   coefficients, only on which qubits they act on and how. Every $V_j$'s own
   Pauli-term coefficients are real (a Hermitian generator decomposed in the
   real $\{I,X,Y,Z\}^{\otimes n}$ basis always has real coefficients), so
   there is no analogue of the ancilla pipeline's hop-term phase-mod-$\pi$
   condition (that condition exists there only because non-Hermitian jump
   operators get encoded via a complex-coefficient $X$/$Y$ ancilla-rotation
   trick this pipeline has no need for) -- pure Pauli commutation is already
   the exact fusion criterion, independent of the random $\Delta W_j$ values.
   Checked directly, not just argued: pooling all 25 generators' Pauli terms
   (105 unique strings) and partitioning by
   `../nested_trotter_grouping.py`'s existing greedy conflict-graph
   coloring (reused unmodified -- its phase-mod-$\pi$ check is simply always
   satisfied for real coefficients, so it behaves as pure-commutation
   grouping here) gives 16 mutually-commuting groups (sizes 4-12), and this
   grouping is identical for every trajectory by construction. See
   `../../trajectory_based_sim/testing/plot4_group_trotter_convergence.py`'s
   docstring for the derivation and the validation that reconstructing the
   25 generators from Pauli dicts reproduces the trusted dense construction
   to $\sim10^{-16}$.

## Numerical validation to date (as of 2026-09-29)

Before any Qualtran gadget existed, the physics and the Trotter/sampling
error structure were checked numerically at the dense-matrix level, in
`../../trajectory_based_sim/testing/`. This is what the pipeline architecture
below is built against:

- **`plot1_trajectory_convergence.py` / `plot2_trajectory_vs_dt.py`**
  (pre-existing): convergence of the *unsplit* scheme
  ($U=\exp[-i(H_0\Delta t+\sum_j\Delta W_jV_j)]$, one joint exponential, no
  internal splitting at all) in trajectory count $N$ and in $\Delta t$.
- **`plot3_lie_eq8_verification.py`**: verified white_noise_trotter_1.pdf's
  Proposition 1 (the noise-averaged Lie channel equals the deterministic
  Lindblad splitting $e^{\Delta t\mathcal L_H}e^{\Delta t\mathcal D}$) *and*
  its Eq. (7) non-commutator correction, on the real 25-generator dissipator
  (not just the note's single-qubit toy model). Confirmed the 25 generators
  pairwise do not commute (100% of pairs, typical commutator norm
  $\sim0.4$), so Eq. (7)'s correction is real; resolved it against Monte
  Carlo at $\Delta t\sim0.03$–$0.07$ s ($\sim3$–$8\sigma$), invisible (as
  expected, $O(\Delta t^2)$) at this project's own $\Delta t^\ast\approx
  9.3\times10^{-5}$ s scale.
- **`plot4_group_trotter_convergence.py`**: introduced and validated the
  cross-generator pooled/grouped Trotterization of the anisotropic part
  described in point 4 above; single-step error against the fully exact
  channel resolves into a genuine, $N$-independent systematic plateau
  (not just sampling noise) at $\Delta t\gtrsim0.5/J_\text{max}$, growing
  roughly linearly in $\Delta t$ (first-order behavior, as expected since
  the 16 groups don't commute with each other).
- **`plot5_exact_fid_spectrum.py` / `plot6_exact_fid_spectrum_refined.py`**:
  exact (no approximation at all) FID and its spectrum, used to pick a
  sane $(\Delta t,T)$ operating point before costing anything approximate.
  First pass ($\Delta t=1/J_\text{max}$, $T\approx1$s) showed a dominant
  spectral feature that looked like an alias of the $203.41$ Hz coupling;
  halving $\Delta t$ and extending $T$ to 2 s (the pipeline's current
  working point) showed that feature was actually genuine (a many-body ZULF
  transition, not reducible to a single bare $J$-coupling) — a real
  correction to an initial hypothesis, not a quiet drop. Validated
  parameters: $\Delta t=(1/J_\text{max})/2\approx2.204\times10^{-3}$ s,
  $T\approx2$ s ($N_t=907$ points, $M=906$ steps), decay complete to
  $\lesssim2\%$ of the initial amplitude by $T$.
- **`plot7_group_trotter_fid_convergence.py`**: the first genuine multi-step
  ($M=906$) trajectory-averaged FID, at the validated $(\Delta t,T)$ above,
  $N\in\{100,200,400\}$ trajectories (one stream drawn once, cumulative-mean
  subsets, not independent re-draws). Visually near-indistinguishable from
  exact even at $N=100$; FID/spectrum RMS error shrinks monotonically but
  slower than pure $1/\sqrt N$ (consistent with a real residual systematic
  floor under the shrinking statistical noise). Notably, the accumulated
  error over 906 steps was over an order of magnitude *smaller* than
  naively multiplying `plot4`'s single-step bias by 906 would predict --
  real cancellation across steps/trajectories, not simple linear
  accumulation. A useful, non-obvious data point for the error-budget work
  still to come: a naive worst-case per-step-times-$M$ bound would be very
  pessimistic here.

## Pipeline architecture (draft, 2026-09-29)

Mirrors `../pipeline_plan.md`'s layering (input spec -> algorithmic
primitive -> error budget composition -> resource counting). Reused pieces
carry over unmodified; new pieces are listed with what they need to do.

**Reused unmodified:**
- `../molecule_operators.py` — Hamiltonian/molecule, external field, $\tau_c$
  all already flow through `load_molecule_operators(mol_id, b_vec, tau_c)`.
- `../coherent_trotter_step.py` — the $H_0$ step gadget.
- `../rotation_synthesis_strategy.py` — $\varepsilon_\text{gate}$-parameterized
  rotation synthesis, applies to every rotation in the new gadget too.
- `../coil_readout.py` — X-/Y-basis readout of the (assumed, for now) target
  observable coil; independent of how the dynamics is compiled.
- `../state_prep.py`'s `build_null_state_prep` — placeholder pure initial
  state, same caveats as in `../pipeline_plan.md` (not a validated stand-in
  for the real initial ensemble; "details of initial state" explicitly
  deferred per this phase's scope).

**Built and validated (2026-09-29):**
1. **`hermitian_generators.py`** — the Hermitian-generator + pooling module,
   promoting the ad hoc construction in
   `../../trajectory_based_sim/testing/plot4_group_trotter_convergence.py`
   into a proper reusable module analogous to `../molecule_operators.py`/
   `../nested_trotter_grouping.py`'s role in the ancilla pipeline.
   `build_anisotropic_data(mol_ops)` returns the $V_j$ Pauli dicts, the
   pooled unique-term list (105 for Gemcitabine), the coefficient matrix,
   and the grouping (16 mutually-commuting groups). Validated against the
   trusted dense construction to $2.4\times10^{-16}$.
2. **`anisotropic_trotter_step.py`** — the anisotropic-part step gadget, the
   trajectory-based analogue of `../nested_trotter_step.py` with no ancilla
   dilation/dispatch at all: one concrete trajectory's numeric $\Delta W_j$
   draws feed 105 individual `apply_pauli_exponential` calls directly on the
   system register (conservative baseline, see Open questions item 1 --
   not 16 group-fused gates). Validated: exactly 5 qubits (no ancilla,
   confirmed on an actual Qualtran circuit, resolving point 1's pending
   check above), $105\times44=4620$ T at $\varepsilon_\text{gate}=10^{-9}$
   matching a hand count exactly, and the circuit's tensor-contracted
   unitary matches the dense sequential-group construction to
   $1.6\times10^{-15}$.
3. **`trotter_scheme.py`** — the `TrotterScheme` abstraction (`LieScheme`,
   `StrangScheme`), mirroring `RotationSynthesisStrategy`'s swappable-strategy
   pattern. Both validated against their expected dense assembly to
   $\sim2\times10^{-15}$, both confirmed to need exactly 5 qubits. Surfaced a
   genuine, non-obvious finding in the process (not a bug): because
   $\Delta t=(1/J_\text{max})/2$ is tied to $J_\text{max}$ by construction,
   the F0-C0 edge's own three coherent-step terms land *exactly* on special
   Rz angles -- $\pi/2$ (a free Clifford/S-gate) at full $\Delta t$
   (LieScheme), $\pi/4$ (the T-gate's own native angle, 1 T instead of the
   generic 44) at half $\Delta t$ applied twice (StrangScheme). A naive
   $n_\text{rotations}\times44$ hand count would silently be off by 132 T
   (Lie) / 258 T (Strang); the self-test now accounts for this exactly.
4. **`traj_full_pipeline_estimate.py`** — the FID-depth resource-aggregation
   layer and full assembly (state prep -> $M$ Trotter steps, each with fresh
   $\Delta W_j$ -> readout), mirroring `../full_pipeline_estimate.py`'s
   structure. Total cost per trajectory is
   $\sum_{m=1}^{M}(\text{state\_prep}+m\times\text{step}+\text{readout})=
   M(\text{state\_prep}+\text{readout})+\text{step}\times M(M+1)/2$, times
   $N$ trajectories, reported separately for X-/Y-readout (never summed).
   Additionally checks, at the fully-assembled-chain level (not just one
   isolated step): $M$-linearity ($M_\text{test}=1,2,3$, exact for both
   schemes) and trajectory-independence (two different seeds, exact
   `QECGatesCost`/`QubitCount` match for both schemes) -- resolving Open
   questions item 2 below. At the validated $(\Delta t,T)=((1/J_\text{max})/2,
   2\text{s})$ operating point ($M=906$), one trajectory's full FID costs
   $\approx2.48\times10^9$ T (Lie) / $\approx3.06\times10^9$ T (Strang);
   $N=100$ trajectories, $\approx2.48\times10^{11}$ T (Lie). **Not yet
   comparable to the ancilla pipeline's $\approx1.36\times10^9$-T baseline**
   -- that number is one fixed-time single-shot estimate at
   error-budget-*derived* $(\Delta t,M)$; this one is a full-FID,
   many-trajectory total at *assumed* hyperparameters. The two become
   honestly comparable only once the deferred error-budget work produces
   matched hyperparameters for both at the same target precision and
   acquisition scenario.

## Open questions / gaps to close

1. **Does the 16-group pooling reduce gate count, or only tighten the error
   bound (as `../nested_trotter_grouping.py`'s own grouping does in the
   ancilla pipeline, explicitly not changing that pipeline's gate-counting
   model)?** Mutually-commuting Pauli terms factor exactly into a sequential
   product of their individual exponentials regardless of support overlap
   ($e^{i\sum_k a_kP_k}=\prod_ke^{ia_kP_k}$ whenever the $P_k$ commute) --
   that's already enough to justify treating a group as safe to reorder
   freely, but it does *not* by itself reduce gate count below "one rotation
   gadget per pooled term" (105 of them). A genuine reduction would need a
   simultaneous-diagonalization circuit for a whole commuting clique (a known
   technique from VQE measurement-reduction, unimplemented anywhere in this
   codebase). **Decision for the first pass (see Decision log): conservative
   baseline -- 105 individual term-rotations per step, grouping used only for
   the error bound.** Revisit as an optimization, not a correctness question.
2. ~~**Verify, don't assume, trajectory-independence and $M$-linearity at the
   Qualtran level.**~~ **Resolved 2026-09-29.** `traj_full_pipeline_estimate.py`
   built $M_\text{test}=1,2,3$ literal chains (fresh $\Delta W_j$ every step)
   for both `LieScheme` and `StrangScheme`: T-count scales exactly linearly
   in $M_\text{test}$ and `QubitCount` stays constant at 5 (no transient
   ancilla at all, unlike the dispatch pipeline's $N_\text{sel}-1$ flag
   qubits) for both. Trajectory-independence checked directly, not just
   argued from the Pauli-commutation structure: the same $M_\text{test}=3$
   chain rebuilt with an independent seed gives byte-for-byte identical
   `QECGatesCost`/`QubitCount` for both schemes.
3. **The full sampling/shot structure is a triple product, and this pipeline
   only ever computes one factor of it.** $N$ trajectories $\times$ $M$ FID
   depths $\times$ (shots-per-realization needed to beat down quantum
   measurement noise, still explicitly deferred in `../pipeline_plan.md`
   too) $\times$ 2 readout variants. Resolving $N$ and the shot count both
   need the error-budget work this phase is explicitly not doing yet.
4. **State-prep unification, later.** The "trajectory" framing here is a
   natural place to eventually revisit `../pipeline_plan.md`'s original (and
   later superseded, for that pipeline's baseline) idea of sampling the
   *initial state* as a pure-state ensemble reproducing $\rho_0$ -- since
   this pipeline is already built around stochastic sampling for the noise,
   unifying the two rather than keeping two unrelated placeholders may be a
   natural next step. Not now, per this phase's scope (initial-state details
   explicitly deferred).
5. **Error-budget-to-hyperparameter derivation.** Explicitly out of scope for
   this phase (see Design constraints) -- $N$, $M$, $T$ (hence $\Delta t$),
   $\varepsilon_\text{gate}$, and the Trotter order are all assumed given as
   external inputs for now, pending the collaboration dependency mentioned
   there.

## Design constraints

Same as `../pipeline_plan.md`: modular in the pulse sequence, and
platform-agnostic (algorithmic-gadget level — Hamiltonian-simulation
queries, Trotter steps, ancilla counts — deliberately without compiling
down to a QEC code or physical error rates).

**Scope for this phase (set 2026-09-29):** this pipeline's error budget and
its allocation across sources of error depend on a collaboration whose
details are still being worked out, and are explicitly out of scope here.
The Qualtran machinery is built assuming the resulting hyperparameters --
number of trajectories $N$, number of Trotter steps $M$, longest evolution
time $T$ (hence $\Delta t=T/M$), target gate-synthesis precision
$\varepsilon_\text{gate}$, and Trotter order -- are externally supplied, not
derived here. Once the error-budget work lands, it plugs into this layer the
same way `../error_budget_solver.py`/`../optimal_error_budget_split.py`
plug into the ancilla pipeline's already-built gadgets.

## Decision log

- **Per-step anisotropic gadget: group-Trotterized (pooled, cross-generator
  grouping), not the coarser single-joint-exponential Lie scheme.**
  (2026-09-29.) Both were validated numerically
  (`plot3_lie_eq8_verification.py` vs. `plot4_group_trotter_convergence.py`
  / `plot7_group_trotter_fid_convergence.py`); the group-Trotterized version
  is the one actually being built into Qualtran gadgets, since it's the more
  circuit-realistic of the two (a single joint exponential of a many-term
  Pauli sum isn't itself a primitive quantum gate; the group-Trotterized
  version is already expressed as a sequence of individually-compilable
  Pauli-string rotations).
- **Gate-counting baseline for the 16 pooled groups: conservative.**
  (2026-09-29.) Apply every one of the 105 pooled individual Pauli-term
  rotations sequentially; grouping is used only to justify that reordering
  terms within a group is exact (not to claim a joint-diagonalization
  gate-count reduction, which is unimplemented and unverified -- see Open
  questions item 1).
- **First-order Lie splitting is the working baseline, behind a
  `TrotterScheme` abstraction so Strang is a config swap.** (2026-09-29.)
  Not derived from an error budget (that derivation is out of scope this
  phase); chosen so there is a concrete scheme to build gadgets against now,
  with the explicit requirement (stated when this phase's scope was set)
  that switching to second order later must not require rewriting the
  per-step gadget.
- **An FID is $M$ separate circuit executions of increasing depth, not one
  $M$-step circuit with intermediate readouts.** (2026-09-29.) A real
  circuit cannot non-destructively checkpoint mid-execution; resource
  totals for a full FID must sum
  $\text{state\_prep}+m\times\text{step}+\text{readout}$ over
  $m=1,\dots,M$, not just cost the single $m=M$ circuit.
- **Hyperparameters ($N$, $M$, $T$, $\varepsilon_\text{gate}$, Trotter order)
  are assumed externally given for this phase.** (2026-09-29.) The
  error-budget derivation that would produce them depends on a
  collaboration still in progress; see Design constraints.
- **All four Qualtran gadgets built and independently validated the same
  day.** (2026-09-29.) `hermitian_generators.py`, `anisotropic_trotter_step.py`,
  `trotter_scheme.py`, `traj_full_pipeline_estimate.py` -- see Pipeline
  architecture above for what each does and what was checked. Two real
  (non-blocking) implementation bugs caught and fixed along the way: a numpy
  complex-to-float cast that raises on current numpy versions (`.real` was
  missing on the self-conjugate-generator branch), and a
  `build_coil_readout` return-value unpacking mismatch copied from the
  ancilla pipeline's 3-value signature.
- **Renamed `full_pipeline_estimate.py` to `traj_full_pipeline_estimate.py`.**
  (2026-09-29.) It shared its exact filename with `../full_pipeline_estimate.py`
  (the ancilla pipeline's own script) -- a real collision, not a hypothetical
  one: caught it directly when a one-off diagnostic's `from
  full_pipeline_estimate import build_chain, cost_of` silently resolved to
  the *ancilla* pipeline's module instead (whichever directory happened to
  be earlier in `sys.path`, itself an accident of which module's own
  `sys.path.insert(0, QRE_DIR)` ran last), failing loudly on a signature
  mismatch rather than a quieter wrong-result failure. Renaming removes the
  collision outright rather than relying on import order staying accidentally
  correct.

## Status

Every gadget this pipeline named now exists, is independently verified, and
has been assembled into one end-to-end resource total for truncated
Gemcitabine, for both `LieScheme` and `StrangScheme`: state prep ->
$M$ x (coherent + anisotropic Trotter step, fresh $\Delta W_j$ every step) ->
readout, summed over $M$ FID depths and $N$ trajectories. This is the
pipeline's first genuine complete baseline number, not a final validated
resource estimate -- every caveat already logged above still applies
verbatim: hyperparameters are assumed, not derived from an error budget
(Design constraints); state prep is the null placeholder, not a validated
$\rho_0$ substitute (Open questions item 4); the shot-count factor is still
out of scope (Open questions item 3); and the total is not yet comparable to
the ancilla pipeline's own baseline (Pipeline architecture item 4). Next
step: the error-budget work (deferred to the collaboration mentioned in
Design constraints) is what would turn the assumed hyperparameters into
derived ones and make that comparison meaningful.
