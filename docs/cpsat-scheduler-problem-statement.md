# Joint tile-scheduling + eviction for a mixed-precision GEMM accelerator — problem statement

*Self-contained problem definition for soliciting help (from humans or other AIs). No project-specific jargon assumed.*

---

## 0. TL;DR

We schedule a tiled matrix multiply (GEMM) onto an accelerator with a small shared on-chip SRAM in front of DRAM. We must jointly decide **(a) the order** in which the small tile-multiplications ("cubes") run and **(b) which operand tiles to keep resident vs. evict** at each step, to **minimize total off-chip DRAM traffic** subject to capacity and a hard "no-stall" double-buffering constraint. Three features make it genuinely hard: **variable tile sizes** (mixed precision), **output partial-sum write-back** (spill/reload of accumulators), and a **per-step rate constraint** (each fetch must hide under the previous step's compute). Small instances are solved to proven optimality three independent ways; the exact methods do not scale, and we want a good large-scale heuristic **plus a tight, honest lower bound** to certify how far from optimal it is.

---

## 1. Setting (why this problem exists)

A systolic-array accelerator multiplies `C = A x W` (activations x weights) in `32x32` spatial tiles. Operands stream from DRAM through a single shared on-chip SRAM scratchpad of fixed capacity. DRAM traffic dominates energy, so the scheduler's job is to exploit on-chip **reuse** (keep a tile resident and reuse it instead of re-fetching) under a tight capacity budget. The weights use **mixed precision** (each weight tile is 2-8 bits/element), so weight tiles have **different sizes**. Output partial sums accumulate over the K dimension and may have to be **spilled** to DRAM and reloaded if they cannot stay resident.

This is the classic "DNN accelerator dataflow/mapping" problem, but with three twists (Section 5) that break the usual clean formulations.

---

## 2. Inputs and notation

- Matrices tiled by `TILE = 32`. Let `MT = M/32`, `KT = K/32`, `NT = N/32` (all integers).
- A **cube** is one `32x32x32` tile-multiply, indexed `(m, k, n)` with `0 <= m < MT`, `0 <= k < KT`, `0 <= n < NT`. There are `T = MT * KT * NT` cubes; each must be executed exactly once.
- Each cube `(m,k,n)` needs **three operand tiles simultaneously resident** to execute:
  - **A tile** `(k, n)` — activation. Size `= 32 * 32 * a_bits` bits, where `a_bits in {2,4,8}` is a fixed activation precision. Reused across `m` (all `MT` cubes with the same `(k,n)` share it).
  - **W tile** `(m, k)` — weight. Size `= 32 * 32 * w_bits[m][k]` bits, where `w_bits[m][k] in [2,8]` is the per-tile **mixed precision** (this is the source of **variable tile sizes**). Reused across `n`.
  - **C tile** `(m, n)` — output partial sum. Size `= 32 * 32 * 32` bits (FP32, fixed). **Accumulated over k**: the `KT` cubes `(m, *, n)` all add into the same C tile.
- On-chip capacity: `cap` bits. At every step the total size of resident tiles must be `<= cap`.
- DRAM: `eff_bw` bits per on-chip compute-cycle (effective bandwidth, already converted into the compute clock domain).
- Compute time of a cube (bit-serial in the weights): `compute(m,k,n) = c * 32 * w_bits[m][k]` on-chip cycles, where `c > 0` is a constant. (Higher-precision weights take proportionally longer to compute.)

All sizes/quantities are known up front (this is an **offline** problem — the full trace is known in advance).

---

## 3. Decision variables

1. **Execution order**: a permutation `pi` of all `T` cubes — `pi(0), pi(1), ..., pi(T-1)`. *Not* restricted to loop-nest orders; any permutation is allowed. (Empirically the optimum is often a non-loop order, e.g. serpentine or per-row/column, so restricting to loop nests is suboptimal — verified by exhaustive search on small instances: up to +25% worse.)
2. **Residency / eviction**: for each step, which tiles are on-chip. Equivalently, at each step you may load missing tiles from DRAM and evict resident ones. A tile that is evicted and later needed again must be re-loaded.

A schedule is the pair (order, per-step eviction choices).

---

## 4. Objective and cost model (this is the exact, single source of truth)

**Minimize total off-chip DRAM traffic in bits** (energy is this times a positive constant, so it is an argmin over bits). Traffic is charged as follows, evaluated step by step as cubes run in order `pi`:

For the cube running at a step, **load** any of its three tiles that are not currently resident, and you may first **evict** resident tiles to make room. Charges:

| Event | Charge |
|---|---|
| Load an **A** or **W** tile (not resident -> resident) | its size (charged on **every** load: first touch *and* any reload after eviction) |
| Load a **C** tile when its accumulation counter `> 0` | its size (reload of a previously spilled partial sum) |
| Load a **C** tile when its accumulation counter `= 0` | **0** (fresh zero-initialized tile: occupies space, but no DRAM read) |
| Evict a **C** tile when `0 < counter < KT` | its size (**spill** the partial sum to DRAM) |
| Evict a **C** tile when `counter = 0` or `counter = KT` | **0** (no data yet / already complete) |
| Evict an **A** or **W** tile | **0** (read-only inputs) |

- **Accumulation counter** of C tile `(m,n)` at a given step = the number of cubes `(m, *, n)` already executed **strictly before** this step (ranges `0..KT`; reaches `KT` when the C tile is fully accumulated).
- The **final write** of each completed C tile to DRAM (`MT*NT` writes total) is a **schedule-invariant constant** and is **excluded** from the objective (it does not change the argmin).
- Consequence: an all-resident schedule (nothing ever evicted) still pays the **first-touch floor** = (sum of sizes of all distinct A and W tiles) — each loaded once. This floor is a valid, unavoidable lower bound on the objective.

**Objective** = (sum of all A/W loads) + (sum of all C reloads) + (sum of all C spills), in bits.

---

## 5. Constraints

1. **Co-residency**: at the step a cube runs, its A, W, and C tiles are all resident.
2. **Capacity**: at every step, `sum of resident tile sizes <= cap`. *(Modeling choice: capacity is charged on the post-load resident set per step; we do **not** separately reserve a "double-buffer peak" for an in-flight prefetch on top of the working set. This is a deliberate simplification — see Open Questions.)*
3. **C accumulation semantics**: the `KT` cubes `(m,*,n)` write into the same C tile; partial sums must be preserved (kept resident) or spilled+reloaded (charged) between contributions; reordering changes which partials must be spilled.
4. **stall = 0 (hard double-buffering / rate constraint)**: for every step `i >= 1`, the data moved to set up step `i` must hide under the compute of step `i-1`:
   `traffic_bits(i) / eff_bw  <=  compute(cube at step i-1)`,
   where `traffic_bits(i)` = the bits loaded+spilled at step `i` (the same quantity charged in the objective), and `compute(.)` is from Section 2 (so the budget scales with the **previous** cube's weight precision). Step `i = 0` (the very first cube, "fill") is exempt. **This is a per-step RATE constraint, not a total-traffic budget** — it couples each step's traffic to the previous cube's choice, which is what ties ordering and eviction together.

A schedule is **feasible** iff it satisfies (1)-(4). Infeasibility is reported, not silently patched (e.g., "no stall=0 schedule exists; the tightest structural schedule still stalls X cycles").

---

## 6. Tiny worked example (to anchor intuition)

`MT=1, KT=2, NT=2` -> `T = 4` cubes: `(0,0,0), (0,1,0), (0,0,1), (0,1,1)`.
- A tiles: `(k,n)` in {(0,0),(1,0),(0,1),(1,1)} — 4 tiles.
- W tiles: `(m,k)` in {(0,0),(0,1)} — 2 tiles (sizes differ if `w_bits[0][0] != w_bits[0][1]`).
- C tiles: `(m,n)` in {(0,0),(0,1)} — 2 tiles, each accumulated over `k=0,1`.

Order `(0,0,0) -> (0,1,0) -> (0,0,1) -> (0,1,1)` finishes C(0,0) before starting C(0,1): C(0,0) stays resident across its two contributions (no spill), then is completed (final write, excluded) and freed; reuse of W tiles depends on capacity. A different order that interleaves the two C groups may be forced to spill a partial C(0,0) (charged) to make room, then reload it (charged). The scheduler trades A/W-reuse savings against C-spill costs, under the per-step capacity and stall=0 limits.

---

## 7. Why it is hard (relation to known problems)

The cost+objective are **linear** and the whole thing is a clean **0/1 integer linear program** (we have built one). The difficulty is **not** writing it down; it is that the integrality gap is large and the problem is provably intractable:

1. **Variable tile sizes (mixed-precision W)** -> the per-step capacity is a knapsack/bin-packing-flavored constraint, and the caching subproblem is **variable-size ("weighted") caching = NP-hard**. (Uniform-size caching is polynomial via Belady's furthest-in-future.)
2. **Output write-back (C spill/reload)** -> **writeback-aware caching**, proven **NP-complete and APX-hard even at unit cost** (no PTAS unless P=NP); Belady is only `(omega+1)`-competitive and that bound is tight [Beckmann, Gibbons, Haeupler, McGuffey, APOCS 2020]. The structural reason: writes create *overlapping* "competing intervals" (between consecutive writes), unlike the disjoint read-reuse intervals of classic caching.
3. **stall = 0 is a per-step rate constraint** that couples consecutive steps and depends on the previous cube's precision -> ordering and eviction cannot be decomposed (solving order first, then eviction, is provably suboptimal here).
4. **The order is a decision variable** and the optimum is often a non-loop permutation -> cannot enumerate loop nests.

Time-indexed / assignment ILP formulations of (1)-(4) have a **weak LP relaxation** (fractional solutions spread a cube across steps, evict fractionally, accumulate fractionally), so branch-and-bound does not close even medium sizes (see Section 8).

---

## 8. What has been tried / what is known

- **Small instances (`T <= ~12`) are solved to PROVEN optimality three independent ways and all agree bit-exactly**: (i) exhaustive DP over all orders x all eviction subsets; (ii) A* best-first joint search; (iii) CP-SAT (0/1 ILP with lazy clause generation). This validates the cost model.
- **The exact methods do not scale.** A* exhausts memory (open list blows up) at `T >= 32` under capacity pressure. CP-SAT does **not** prove optimality at `T = 32/64`: its lower bound is essentially the (weak) LP-relaxation bound and sits **below even the first-touch floor**, so the certified "honest gap" is 80-400x and uninformative; without a good warm start its incumbent is worse than a simple structural baseline. (Diagnosis: the formulation lacks strengthening cuts; e.g. "every used tile is loaded at least once" is not enforced in the relaxation.)
- **Real target workloads are large**: transformer GEMMs (e.g. DeiT, Qwen-family) give `T = 10^3 - 10^6`, with on-chip footprint `>> cap` so eviction is unavoidable. Proven-optimal is out of reach (theory + scale).
- The **intended** large-T deliverable was never "proven optimal"; it is **a good heuristic schedule + an honest lower bound** that certifies the heuristic's worst-case gap.

---

## 9. What we want help with

1. **Tightest tractable lower bound** on the optimal off-chip traffic for the large-T regime, to certify a heuristic's honest gap. Known candidates: the first-touch floor; a read-only A/W bound via min-cost flow / interval LP (relax write-back + stall); Lagrangian relaxation of the capacity + stall=0 coupling. **Can any of these be made tight on structured GEMM traces (not adversarial)?** What is the best bound that is still computable at `T = 10^6`?
2. **Best heuristic** for the joint (order + eviction) schedule with variable-size tiles + write-back + the stall=0 rate constraint. (We are considering: analytical loop-order-class scoring with closed-form mixed-precision weight cost, then windowed eviction using a write-back-aware policy such as Writeback-Aware Landlord for the C tiles and Belady/GreedyDual for read-only A/W.)
3. **Stronger formulations** that might close *medium* sizes (`T = 32-64`) or tighten the bound: cutting planes / valid inequalities, extended (network-flow) formulations, Dantzig-Wolfe / column generation, Benders, or a different decomposition that respects the order-eviction coupling.
4. **Exploitable structure** we may be missing that yields a polynomial special case or a provably near-optimal scheme: the GEMM reuse geometry (A reused over `m`, W over `n`, C reduced over `k`), the fact that `K, N` are multiples of 32, the bounded precision set `{2,4,8}` (and `w_bits in [2,8]`), or the bounded number of distinct tile sizes.

### Constraints on proposed solutions
- **Offline** (full trace known in advance), **deterministic**, single-objective (minimize off-chip bits as defined in Section 4).
- A heuristic must come with an **honest gap** (a computed lower bound), not just "it works well in practice."
- Keep the **stall=0 rate constraint** and the **write-back C semantics** exactly as in Sections 4-5; simplifying them away changes the problem (and was the failure mode of earlier flow-only attempts).

---

## 10. One-line restatement

Offline, minimize DRAM bits for an ordered sequence of `32^3` GEMM cubes sharing a capacity-bounded scratchpad, where weight tiles have variable (mixed-precision) sizes, output tiles are accumulators that cost a write-back when spilled mid-accumulation, and each step's fetch must finish within the previous cube's (precision-dependent) compute window — jointly choosing the cube order and the per-step eviction set. Exact for tiny sizes; need a scalable heuristic + a tight honest lower bound for `T` up to `10^6`.
