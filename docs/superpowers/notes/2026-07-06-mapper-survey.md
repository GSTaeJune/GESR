# Mapper / mapping-space-search survey (2021-2026, top-tier venues)

Date: 2026-07-06
Scope: the "mapper" problem — given a fixed accelerator and a layer/GEMM, find loop
tiling + loop order + spatial mapping minimizing energy/latency. Judged against THIS
project's profile: one on-chip level, three dedicated ping-pong buffers (W/A/O),
32x32 bit-serial systolic array, dense GEMM only, mapspace ~10^3-10^5 (exhaustively
enumerable), final scoring by our own analytical model (mapper = generator only).

Venue policy: top-tier = ISCA / MICRO / ASPLOS / HPCA / MLSys / DAC / IEEE TC /
TCAD / TODAES. Tools outside that set are kept only where load-bearing and marked
[non-top-tier].

## 1. Tool table

| Tool | Venue/Year (verified) | Search algorithm | Cost backend | Claimed win regime | Hard deps | OSS / maintenance |
|---|---|---|---|---|---|---|
| Timeloop mapper (baseline) | ISPASS'19 [non-top-tier venue, but the de-facto standard evaluator] | pruned-exhaustive / random / hybrid threads | own analytical model (+Accelergy) | reference model+mapper; exhaustive covers small spaces completely | none (C++); v4 = timeloopfe | github.com/NVlabs/timeloop, actively maintained (v4 "Ruby") |
| CoSA | ISCA 2021 | one-shot MIP (single solve, no iteration) | MIP optimizes 3 PROXY objectives (buffer utilization, compute, NoC traffic, weighted sum); final eval on Timeloop + NoC sim | multi-level hierarchies (Simba-like 5-level + mesh NoC) where random/hybrid search wastes samples; win = time-to-solution (90x) and quality vs BUDGETED Timeloop hybrid (1.5x perf, 22% energy vs 16K-sample search) | Gurobi (license) | github.com/ucb-bar/cosa; low activity |
| Mind Mappings | ASPLOS 2021 | gradient descent on a trained differentiable surrogate of the cost model | own surrogate (trained per problem class); ground truth from an analytical model | huge non-convex mapspaces; win = quality at fixed sample budget vs SA/GA/RL (1.29-1.76x EDP) | surrogate training data + PyTorch | github.com/kartik-hegde/mindMappings; stale |
| GAMMA | ICCAD 2020 [not in the approved venue list — kept as commonly-cited baseline] | genetic algorithm, domain-aware operators | MAESTRO | large spaces, flexible accelerators; win = quality vs other black-box heuristics at equal samples | MAESTRO | github.com/maestro-project/gamma; stale |
| HASCO | ISCA 2021 | Bayesian opt (HW) + heuristic/Q-learning (SW), TVM tensorize codegen | own + TVM measurements | HW/SW co-design (accelerator params + schedule jointly) | TVM stack | github.com/pku-liang/HASCO; stale |
| ZigZag | IEEE TC 2021 (vol 70 no 8) | per-engine: exhaustive loop-order (LOMA) or SA (SALSA) over an ENLARGED space (uneven mapping: different tensors cut at different memory levels) | own analytical model, per-operand (W/I/O) memory hierarchies | single/few-level scratchpad accelerators, per-operand dedicated buffers; joint arch-mapping DSE | none (Python) | github.com/KULeuven-MICAS/zigzag; actively maintained |
| LOMA / SALSA (ZigZag engines) | LOMA: AICAS 2021, SALSA: ISVLSI 2023 [both non-top-tier venues] | LOMA: loop-order enumeration + memory allocation; SALSA: exhaustive+simulated-annealing dual engine | ZigZag model | SALSA claims 7.6-11.9% lower energy than Timeloop/LOMA — but vs Timeloop's BUDGETED hybrid search + a different (larger, uneven) mapspace, not vs exhaustive on the same space | none | inside ZigZag repo |
| Sparseloop | MICRO 2022 | not a mapper — modeling framework (extends Timeloop) | own (Timeloop ecosystem) | sparse tensor accelerators (compression formats, gating/skipping) | none | Timeloop ecosystem, maintained |
| Demystifying Map-Space Exploration for NPUs | IISWC 2022 [non-top-tier, kept for its meta-result] | study, not a tool: apples-to-apples comparison of mapper search techniques | Timeloop/MAESTRO | meta-finding: different search heuristics mostly converge near-optimal on small spaces; differences show up in huge spaces / sample budgets | - | - |
| Explainable-DSE | ASPLOS 2023 | bottleneck-analysis-guided (white-box) acquisition instead of black-box | pluggable bottleneck models | HW/SW codesign with expensive evaluators; win = 47x fewer iterations | none specific | github (ASU MPS-lab); low activity |
| SET | ISCA 2023 | inter-layer (fusion/tiled-accelerator) scheduling space exploration | own | multi-layer scheduling on tiled accelerators — NOT single-GEMM mapping | none | github.com/SET-Scheduling-Project |
| TileFlow | MICRO 2023 | GA + Monte-Carlo Tree Search over a tree-structured fusion mapspace | own tree-based analysis (Timeloop-derived) | FUSION dataflows (multi-op, shared on-chip tiles); huge composite spaces | none | github.com/pku-liang/TileFlow; low activity |
| DOSA | MICRO 2023 | gradient descent on a hand-derived differentiable rewrite of Timeloop's model; one-loop HW+mapping co-search | differentiable Timeloop approximation (+ optional learned correction) | joint buffer-sizing + mapping under sample budgets; 2.8x vs random, 12.6x vs BO at equal samples | PyTorch | github.com/ucb-bar/dosa; research-grade |
| Orojenesis ("Mind the Gap") | ISCA 2024 | not a mapper — computes mapping-INDEPENDENT data-movement lower bounds vs buffer size (incl. fused chains) | own (Timeloop ecosystem) | bounding, gap analysis ("is any mapping better possible?") | none | timeloop.csail.mit.edu/orojenesis |
| FEATHER (+Layoutloop) | ISCA 2024 | HW paper (reconfigurable array + reduction network); Layoutloop = Timeloop extended with layout assessment | Timeloop-based | per-layer dataflow SWITCHING with layout reordering; not a search-algorithm contribution | none | github.com/maeri-project/FEATHER |
| LoopTree | ISPASS 2023 + IEEE TCAS-AI 2024 [neither in approved venue list] | model + exploration of fused-layer design space (tiling/retention/recompute taxonomy) | own | fused-layer accelerator design | none | MIT EEMS |
| Sunstone | ISPASS 2023 [non-top-tier] | analytical/structural scheduling of tensor algebra on spatial accelerators | own | versatility across tensor algebra; scalability | none | research-grade |
| Turbo-Charged Mapper (TCM) | arXiv 2602.15172, Feb 2026 (MIT Sze/Emer group) [PREPRINT — no verified top-tier acceptance yet; not in ISPASS'26 program] | optimality-preserving pruning ("dataplacement" abstraction) + full mapspace search — i.e., provably-optimal pruned-exhaustive | own (Timeloop lineage) | makes OPTIMAL search feasible on big spaces (32 orders of magnitude pruning, 5h -> 17s); on small spaces it just equals exhaustive | none stated | not yet released (as of survey date) |
| Fast-and-Fusiest (FFM) | arXiv 2602.15166, Feb 2026 [PREPRINT] | optimality-preserving pruning for FUSION-aware mapping; reconstructs optimal from partial mappings | own | fusion mapspaces; >10,000x faster than TileFlow/SET, 1.8x EDP vs TransFusion | none stated | not yet released |
| GOMA | arXiv 2603.07962, Mar 2026 [PREPRINT] | integer optimization over a geometric abstraction; claims global-optimal GEMM mapping, O(1) energy eval per candidate | own closed-form analytical energy | GEMM-only spatial accelerators; 2.24-4.24x EDP vs SOTA mappers, 3.8-73.6x faster solve | unclear | no code found |

Sources (primary):
- CoSA: https://arxiv.org/abs/2105.01898 , https://github.com/ucb-bar/cosa , https://dl.acm.org/doi/10.1109/ISCA52012.2021.00050
- Mind Mappings: https://dl.acm.org/doi/abs/10.1145/3445814.3446762 , https://arxiv.org/pdf/2103.01489
- GAMMA: https://github.com/maestro-project/gamma (ICCAD'20)
- HASCO: https://dl.acm.org/doi/10.1109/ISCA52012.2021.00086 , https://github.com/pku-liang/HASCO
- ZigZag: IEEE TC 70(8) 2021, https://github.com/KULeuven-MICAS/zigzag ; SALSA: https://arxiv.org/abs/2304.12931 (ISVLSI'23, IEEE 10168625)
- Sparseloop: https://dl.acm.org/doi/abs/10.1109/MICRO56248.2022.00096
- Demystifying MSE for NPUs: https://arxiv.org/abs/2210.03731 (IISWC'22)
- Explainable-DSE: https://dl.acm.org/doi/10.1145/3623278.3624772 (ASPLOS'23)
- SET: https://dl.acm.org/doi/abs/10.1145/3579371.3589048 (ISCA'23), https://github.com/SET-Scheduling-Project/SET-ISCA2023
- TileFlow: https://dl.acm.org/doi/10.1145/3613424.3623792 (MICRO'23), https://github.com/pku-liang/TileFlow
- DOSA: https://dl.acm.org/doi/10.1145/3613424.3623797 (MICRO'23), https://github.com/ucb-bar/dosa
- Orojenesis: https://dl.acm.org/doi/abs/10.1109/ISCA59077.2024.00021 (ISCA'24), https://timeloop.csail.mit.edu/orojenesis
- FEATHER: https://arxiv.org/abs/2405.13170 (ISCA'24), https://github.com/maeri-project/FEATHER
- LoopTree: https://arxiv.org/abs/2409.13625 (TCAS-AI'24; ISPASS'23 precursor)
- TCM: https://arxiv.org/abs/2602.15172 ; FFM: https://arxiv.org/abs/2602.15166 ; GOMA: https://arxiv.org/abs/2603.07962

## 2. THE KEY VERDICT (for our problem profile)

**No tool in this survey can produce a BETTER mapping than exhaustive enumeration on
our mapspace, under the same cost model. By construction.** Exhaustive enumeration of
a fully-covered space is optimal within that space; every search algorithm (MIP, GA,
MCTS, gradient, SA, surrogate) can at best tie it. The literature is consistent with
this:

1. **Every "quality win" in these papers is vs a BUDGETED or INCOMPLETE baseline,
   never vs completed exhaustive search.** CoSA's 1.5x/22% is vs Timeloop's hybrid
   random search capped at ~16K valid samples of a multi-level Simba-like space
   (arXiv 2105.01898, Sec. VI). SALSA's "7.6% lower energy than Timeloop" is vs the
   same budgeted hybrid search, and partly comes from searching a LARGER space
   (ZigZag's uneven mapping), not from a smarter algorithm on the same space.
   Mind Mappings/GAMMA/DOSA all report quality-at-equal-sample-budget vs other
   heuristics. The IISWC'22 "Demystifying" study makes the meta-point explicit:
   on modest spaces, all reasonable searchers converge to near the same optimum.

2. **The 2026 frontier (TCM, FFM, GOMA) confirms the frame: the research problem is
   making OPTIMAL search fast on huge spaces, not beating exhaustive on small ones.**
   TCM = optimality-preserving pruning so that a full mapspace search finishes
   (5h -> 17s). That is exactly "exhaustive, but faster" — on a 10^3-10^5 space that
   already enumerates in seconds-to-minutes, it buys nothing.

3. **A tool CAN "win" only in two ways that don't apply (or apply against us):**
   - (a) It searches mappings OUTSIDE our enumerated space (e.g., ZigZag's uneven
     mapping, imperfect factors). Our space is small and we define it ourselves for
     our own evaluator; if a mapping class is missing, the fix is to enumerate it,
     not to adopt a heuristic searcher.
   - (b) Its internal cost model disagrees with ours and it "finds" mappings its own
     model likes. Since our pipeline scores with OUR analytical model (bit-serial
     timing, ping-pong drain, project pJ coefficients), a mapper optimizing a foreign
     proxy (CoSA's utilization/traffic MIP, MAESTRO, ZigZag's model) can only be
     as good as the proxy's agreement with our model — i.e., it is strictly a risk,
     never an upside, when exhaustive-under-our-model is affordable.

**Bottom line: enumerate the full (tiling x permutation) space with timeloop-mapper
or a ~100-line generator, score every candidate with our own model, done. That IS the
state of the art for this regime; the SOTA tools exist for regimes we don't have
(multi-level hierarchies, fusion, sparsity, 10^20+ spaces, HW co-design).**

## 3. Anything closer to our HW than Timeloop v4? (Q4)

- **Bit-serial timing / ping-pong drain semantics: no surveyed top-tier mapper models
  either.** Timeloop, MAESTRO, ZigZag all assume word-parallel MACs and
  perfectly-overlapped double buffering. Our external evaluator remains mandatory;
  this is another argument for "mapper = candidate generator only".
- **Per-dataspace dedicated buffers:** ZigZag (IEEE TC'21) is the most natural fit —
  its memory model is per-operand (W/I/O hierarchies can differ), matching our three
  dedicated buffers without Timeloop's bypass directives. But Timeloop v4 expresses
  the same architecture fine via keep/bypass, and we already have it installed and
  verified. Not worth switching for generation-only duty.
- **Precision-scalable PEs:** the ZigZag group (KU Leuven, Verhelst) is the ecosystem
  that has published precision-scalable MAC DSE, and ZigZag exposes per-operand word
  sizes; still, it models precision as bandwidth/word-size scaling, not true
  bit-serial cycle behavior (W_PREC-dependent M-progression). No faithful off-the-shelf
  model exists.
- **GEMM-only:** GOMA (arXiv Mar 2026, preprint) is the only GEMM-specialized
  "global-optimal" mapper found — closed-form energy + integer optimization. Watch it
  for a venue acceptance + code release, but its win is again solve TIME (O(1) eval),
  not quality over exhaustive, and its cost model is its own, not ours.
- **Lower bounds (adjacent, genuinely useful):** Orojenesis (ISCA'24) computes
  mapping-independent data-movement bounds vs buffer size. Not a mapper, but directly
  relevant to this project's separate "tight honest lower bound" thread for the
  MXP_scheduler (tighter than first-touch floor for some regimes; it comprehends
  buffer-limited reuse).

## 4. Sanity check: "CoSA needs Gurobi and optimizes a proxy objective, evaluated on Timeloop" (Q5)

**CONFIRMED on all three counts.**
- Gurobi: repo README instructs obtaining a Gurobi license (free academic) —
  https://github.com/ucb-bar/cosa (`pip install cosa-scheduler` still needs the
  license).
- Proxy objective: the MIP objective is a weighted sum of three heuristic terms —
  buffer-utilization (log-transformed geometric-mean utilization), compute (minimize
  temporal iterations), and NoC-traffic — not exact energy or latency. The paper
  itself notes tuning the objective weights "could further improve" results, i.e.,
  the proxy is acknowledged as approximate (arXiv 2105.01898).
- Evaluation: generated schedules are scored post-hoc with Timeloop (pinned commit
  019f107 in the repo) plus a cycle-exact NoC simulator.
- Context for its wins: one-shot solve, 90x faster time-to-solution and better
  quality than a ~16K-sample budgeted Timeloop hybrid search on a 5-level Simba-like
  hierarchy. None of that transfers to a single-level, fully-enumerable space.

## 5. When to revisit a SOTA search tool

Adopt/re-evaluate a tool only if one of these becomes true:
1. **Mapspace explodes past enumerability** (multi-level on-chip hierarchy added,
   or joint per-layer + cross-layer decisions push candidates past ~10^7): first
   candidate = TCM-style optimality-preserving pruning (watch arXiv 2602.15172 for
   venue + code); CoSA-style MIP only if a Gurobi license is acceptable and a proxy
   objective is tolerable.
2. **Fusion enters scope** (e.g., mapping whole attention chains QK^T -> softmax -> PV
   with intermediate psums kept on-chip): TileFlow (MICRO'23), SET (ISCA'23),
   FFM (arXiv 2602.15166), LoopTree taxonomy; Orojenesis for fused bounds.
3. **HW parameters become searchable again** (buffer sizing, array shape co-design):
   DOSA (MICRO'23) for gradient-based one-loop co-search; Explainable-DSE (ASPLOS'23)
   if each evaluation is expensive.
4. **Sparsity** (weight/activation compression, gating): Sparseloop (MICRO'22).
5. **We need a tighter mapping-independent lower bound** for honest-gap reporting in
   MXP_scheduler: Orojenesis (ISCA'24) — worth reading regardless of mapper choice.
6. **GOMA gets accepted + released**: re-check whether its closed-form GEMM
   formulation can be re-targeted to our cost coefficients; if yes it could replace
   enumeration as the generator (speed/elegance, not quality).
