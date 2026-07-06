# Timeloop smoke test: GEMM accelerator (W/A/O dedicated buffers + 32x32 SA)

Status: **SUCCESS** (2026-07-06). Mapper converges in ~1 minute of wall clock inside WSL.

- Workload: one GEMM, `Z[m,n] += A[m,k] * B[k,n]`, M=224 K=192 N=768 (deit_tiny fc1, padded).
- Arch: DRAM -> three dedicated single-dataspace buffers (W=8KiB@8b, A=8KiB@8b, O=16KiB@32b) -> 32x32 MAC array (SA row=K, col=N, M temporal).
- Result: EDP-optimal mapping found, 100% array utilization, Cycles=32256, Energy=321.03 uJ, 9719 fJ/compute.

## How to run

```
wsl bash -lc '/mnt/c/Users/ptj72/Desktop/Desktop/00project/gemm_sram/buffer_sweep/timeloop/run_smoke.sh'
```

`run_smoke.sh` activates conda env `timeloop`, stages the four input files into
`~/tl-gemm-smoke/` (WSL-native fs), runs `run_smoke.py`, and copies `output/`
back next to the script. `run_smoke.py` is a ~20-line timeloopfe driver:

```python
import timeloopfe.v4 as tl
spec = tl.Specification.from_yaml_files("arch.yaml", "components/smartbuffer_SRAM.yaml",
                                        "components/intmac.yaml", "problem.yaml", "mapper.yaml")
tl.call_mapper(spec, output_dir=OUT, dump_intermediate_to=OUT)
```

Input files (this dir): `arch.yaml`, `problem.yaml`, `mapper.yaml`, `components/{smartbuffer_SRAM,intmac}.yaml`.
Raw outputs of the successful run: `output/` (`timeloop-mapper.map.txt`, `.stats.txt`, `.map+stats.xml`, `.map.yaml`, ERT/ART, `parsed-processed-input.yaml`). Search console log: `mapper_run.log`.

## (a) Which YAML dialect works

**timeloopfe v4** (the "v0.4" spec): YAML with `version: 0.4` and node tags
`!Container` / `!Component` / `!Parallel` / `!Hierarchical`, processed by the
Python front-end and only then handed to the C++ `timeloop-mapper` binary.
Do NOT hand-write classic timeloop v0.3-style `arch/prob/mapspace` files for
this install.

Install quirks:
- The Python module here is `timeloopfe` (standalone package in conda env
  `timeloop`), **not** `pytimeloop.timeloopfe.v4` as in newer accelergy-exercises
  scripts. `import timeloopfe.v4 as tl`.
- The front-end merges + expands everything into
  `output/parsed-processed-input.yaml` (flat v0.4, no tags). That file can be
  fed directly to `timeloop-mapper <file>` for debugging — same trick the prior
  successful run at `~/tl-verify-v4/simple_output_stationary/` used.
- Compound component classes (`smartbuffer_SRAM`, `intmac`) are NOT built in;
  their definitions must be passed as extra input YAMLs (copied verbatim from
  `~/timeloop-install/timeloop-accelergy-exercises/workspace/example_designs/example_designs/_components/`).
- Per-dataspace buffers did **not** need a workaround: a `!Parallel` node with
  three `!Component` buffers, each with
  `constraints: {dataspace: {keep: [X], bypass: [Y, Z]}}`, works as intended.

## (b) How the winning mapping is reported

`output/timeloop-mapper.map.txt` (human-readable loop nest, outer loops at top,
innermost at bottom; bracket numbers = utilized tile size **in words/elements**
for each kept dataspace at that level):

```
DRAM [ A:43008 (43008) B:147456 (147456) Z:172032 (172032) ]
------------------------------------------------------------
| for M in [0:7)

w_buffer [ A:6144 (6144) ]
--------------------------
|   for N in [0:24)

a_buffer [ B:6144 (6144) ]
--------------------------
|     for M in [0:8)

o_buffer [ Z:128 (128) ]
------------------------
|       for M in [0:4)
|         for K in [0:6)

inter_PE_spatial [ ]
--------------------
|           for N in [0:32) (Spatial-Y)
|             for K in [0:32) (Spatial-X)
|               << Compute >>
```

**Extracting per-level tile sizes (m,k,n at a buffer):** a level's tile of a
dimension = product of that dimension's loop bounds at all levels *below* it
(temporal and spatial). For this mapping:

| level | m tile | k tile | n tile | kept dataspace tile | capacity | fits |
|---|---|---|---|---|---|---|
| w_buffer | 8*4=32 | 6*32=192 | (24*32=768) | A: 32*192 = 6144 wd (8b) | 8192 | yes |
| a_buffer | (8*4=32) | 6*32=192 | 32 | B: 192*32 = 6144 wd (8b) | 8192 | yes |
| o_buffer | 4 | (6*32) | 32 | Z: 4*32 = 128 wd (32b) | 4096 | yes |

(Parenthesized dims are not projected by that buffer's dataspace.) These equal
the bracket numbers in map.txt — cross-check passed. Full-problem check:
7 * 24 * 8 * (4*6) * (32*32) = 33,030,144 = 224*192*768 MACs. Cycles =
33,030,144 / 1024 PEs = 32,256 (100% utilization, so no stalls modeled).

**Extracting the temporal permutation:** `output/timeloop-mapper.map.yaml` is
the machine-readable form — one entry per (target, type):

```yaml
- {target: o_buffer, type: temporal, factors: M4 N1 K6, permutation: KMN}
```

`factors` = the loop bounds AT that level; `permutation` is listed
**innermost-first** (leftmost symbol = innermost loop). Unit-factor (X=1) loops
are placeholders — only non-1 factors order meaningfully. E.g. o_buffer
`KMN` + `M4 K6` = `for M in [0:4) { for K in [0:6) }` (K innermost), matching
map.txt. The same info is in `timeloop-mapper.map+stats.xml` (loopnest per
level) if XML parsing is preferred.

## (c) Energy / cycles in the stats output

`output/timeloop-mapper.stats.txt`:
- Per level (`=== mac ===`, `=== o_buffer ===`, ... top of file): `Cycles`,
  `Utilized capacity` (words), `Total scalar accesses`, and `Energy (total)`
  in **pJ**.
- End-of-file `Summary Stats` block — the part to parse for sweeps:

```
Summary Stats
-------------
GFLOPs (@1GHz): 2042.67          <- derived from global_cycle_seconds=1e-9
Utilization: 100.00%             <- spatial array utilization
Cycles: 32256                    <- unit: cycles
Energy: 321.03 uJ                <- unit: microjoules, total
EDP(J*cycle): 1.04e+01
Area: 0.00 mm^2                  <- 0 because aladdin/dummy area plugins

Computes = 33030144
fJ/Compute                       <- unit: femtojoules per MAC, per level
    mac        = 1268.40
    o_buffer   = 36.18
    a_buffer   = 5814.87         <- dominant: B streamed to PEs every cycle
    w_buffer   = 183.09
    DRAM       = 2416.67
    Total      = 9719.20
```

The mapper's search console (stderr) prints one line per improving mapping:
`[thread] Utilization | pJ/Compute | compact-mapping | Cycles` — the compact
mapping string (`L4[ABZ] M7 - L3[A] N24 - ... - L0[] K1 N32Y K32X`) is a
one-line version of map.txt (L4=DRAM ... L0=spatial level, suffix X/Y =
spatial dims).

## (d) Pitfalls hit

1. **Spatial `split` semantics (cost: zero valid mappings, "Fail class:
   Fanout ... mapped fanoutY 1024 exceeds hardware fanoutY 32").** In a
   `!Container` spatial constraint, `permutation` + `split` decide the
   meshX/meshY assignment: dims at permutation index `< split` go to X, the
   rest to Y. timeloopfe prepends any *unlisted* dims to the front of the
   permutation, silently shifting the split point. Writing
   `permutation: [K, N], split: 1` became `MKN / split 1` -> X={M}=1,
   Y={K,N}=1024 -> every mapping violated fanout. Fix: list the **full**
   permutation explicitly: `permutation: [M, K, N], split: 2, factors:
   [K=32, N=32, M=1]`.
2. **Mapper termination knobs are not wall-clock; defaults stall for >10 min
   on this mapspace.** `timeout` = consecutive INVALID mappings per thread,
   `victory_condition` = consecutive non-improving VALID mappings per thread.
   Valid mappings are sparse here (tight buffers + fixed 32x32 spatial), so
   `victory_condition: 100`+`random` (the exercises default) ran silently for
   10+ minutes after plateauing in the first seconds. What converges in ~1 min:
   `algorithm: linear_pruned, victory_condition: 50, search_size: 400,
   timeout: 3000, num_threads: 8`. The best mapping (9719 fJ/compute) is found
   within seconds either way.
3. **Killing the run looks like a silent success.** Wrapping the python driver
   in `timeout ...` produced exit code 0 with a 0-byte `timeloop-mapper.stats.txt`
   and no traceback. Related: timeloopfe's error
   `AssertionError: Could not find cycles in stats` after a completed run means
   "the mapper found no valid mapping" (empty stats), not a parser bug — rerun
   with `mapper.diagnostics: True` to get per-fail-class counts and sample
   mappings.
4. **Mixed word widths at DRAM.** A level has a single `datawidth`; ours is 8b
   at DRAM while Z is physically 32b. Timeloop counts accesses in elements and
   charges DRAM energy at 8b/element, so DRAM energy for Z traffic is ~4x
   undercounted in this smoke config (Z DRAM traffic is small here; acceptable
   for smoke). For the real sweep either give Z its own DRAM-side accounting or
   post-scale Z DRAM energy by 4.
5. **Module name drift**: `import pytimeloop.timeloopfe.v4` (as used by current
   accelergy-exercises `run_example_designs.py`) fails on this install; it ships
   `timeloopfe` as a top-level package.
6. **Run location**: execute in WSL-native fs (`~/tl-gemm-smoke`), not
   `/mnt/c/...` — the mapper writes many small files and /mnt/c (9P) is slow;
   also avoids Windows/WSL file-lock weirdness. `run_smoke.sh` handles the
   staging + copy-back.
7. **Self-kill trap** (infra, not Timeloop): `pkill -f timeloop-mapper` from a
   `bash -lc '...'` one-liner matches the shell's own command line and kills it
   (exit 15). Use a pattern that doesn't appear in the invoking command line.

## (e) Was the 32x32 spatial constraint respected?

Yes. The winning mapping has exactly `for N in [0:32) (Spatial-Y)` and
`for K in [0:32) (Spatial-X)` at `inter_PE_spatial` (the auto-generated dummy
level timeloopfe inserts for a `!Container` with `spatial:`), M spatial = 1,
and `Utilization: 100.00%` (1024/1024 PEs). This matches the RTL mapping
(SA row = K, col = N, M = cycle axis); the mapper additionally confirms
Cycles = MACs/1024 with no idle PEs.
