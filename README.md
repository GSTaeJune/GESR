# gemm_sram (GESR)

Integration of the **MXP mixed-precision bit-serial systolic array** (GEMM compute)
with **parameterized 1RW SRAM banks**, joined by a **Read-Modify-Write (RMW)** datapath.

A column's accumulated GEMM result is dequantized to FP32, added to the prior partial
sum held in SRAM, and written back to the same address — so the SRAM holds the running
tile-wise sum across the K loop.

> Vivado 2024.1 project (`gemm_sram.xpr`, target `xc7vx485tffg1157-1`).
> Synthesis top: `gemm_sram_top`. Simulation top: `gemm_sram_top_tb`.

## Datapath

```
GEMM (INT psum + 9-bit scale) ──► RMW (INT→FP32, FP32 add) ──► sram_1rw_banked (FP32)
                                       ▲                              │
                                       └────── FP32 prior psum ◄──────┘
```

- **GEMM** — MXP 32×32 bit-serial systolic array. Mixed precision per operand
  (A = activation, B = weight) over `{2, 4, 8}` bits. SA dimension mapping:
  `row = K-axis`, `col = N-axis`, `cycle = M-axis`.
- **RMW** — dequantizes INT32 → IEEE-754 FP32 using the combined 9-bit scale,
  then FP32-adds the prior partial sum. HardFloat-based adder
  (`L_CONV = 2`, `L_ADD = 3`, total 5 cycles).
- **SRAM** — `sram_1rw_banked` with `NUM_BANKS = 32`, `BANK_DEPTH = 1024`,
  `PIPELINE = 0`. Active-low `CEB`/`WEB`, active-high `WMASK`.
  Column `j` maps to bank `j` (zero conflicts): `C[m,n] → flat=m*N+n → bank=flat%32, word=flat//32`.

The wrapper (`gemm_sram_top`) is **pure structural** — all controller logic lives in
the testbench, with 32 col-parallel RMW instances and per-column FIFO/drain handling
(`out_fire` is per-column, offset by chain delays).

## Verification status

| Test | Command | Result |
|---|---|---|
| Uniform 9-mode integration | `bash sim/run_integration_sweep.sh` | 9/9 PASS (bit-exact, 16384 elems/mode) |
| Mixed-precision sweep (A ∈ {2,4,8}) | `python sim/runner.py mixed-sweep` | 3/3 PASS (n_diff = 0/16384) |
| RMW vector TB | `bash sim/run_rmw.sh` | 71/71 PASS |
| MXP_Tools unit tests | `cd MXP_Tools && python -m pytest tests/ -q` | 53 PASS |

All HW outputs are verified bit-exact against the `MXP_Tools` Python golden.

## Running simulations

### Python orchestrator (recommended)

```bash
python sim/runner.py                                  # default: mixed-one A=8 (random W) + viz
python sim/runner.py mixed-one --A 8 --uniform 8      # uniform W isolation test
python sim/runner.py mixed-sweep                      # 3-mode mixed-precision (~5 min)
python sim/runner.py integration-one --A 8 --B 8      # single (A,B) uniform mode
python sim/runner.py integration-sweep                # 9-mode uniform sweep (~10 min)
```

Outputs `work/<LABEL>/result.png` (input W/A/prec maps · output C_hw/C_sw/C_fp32 ·
diff · PASS/FAIL stats). Internally wraps the `sim/run_*.sh` scripts.

### XSim batch (bash)

```bash
bash sim/run_integration_one.sh A8_B8 8 8   # one mode end-to-end
bash sim/run_integration_sweep.sh           # all 9 modes serial (~10.5 min)
bash sim/run_rmw.sh                          # full RMW vector TB
```

> **Windows note:** `xsim` is invoked via bash dispatch (not directly) to avoid
> orphaned-process file locks. `-testplusarg "K=V"` must be wrapped as
> `cmd //c "xsim ... -testplusarg \"K=V\""` under Git Bash (the integration
> scripts already do this).

## Repository layout

```
gemm_sram.xpr                          # Vivado 2024.1 project
gemm_sram.srcs/sources_1/
    new/
        gemm_sram_top.v                # Integration wrapper (GEMM + RMW + SRAM, structural)
        GEMM.v                         # MXP TOP wrapper
        RMW.v                          # FP32 RMW unit
        int_to_fp32.v                  # INT32 + 9-bit scale -> IEEE-754 FP32
        fp32_adder.v                   # IEEE-754 FP32 adder (HardFloat-based)
    imports/Desktop/MXP/...            # MXP compute RTL (imported)
    imports/Desktop/sram/rtl/...       # SRAM RTL (imported)
tb/                                    # Testbenches (integration + unit)
sim/                                   # runner.py + run_*.sh sim scripts
third_party/berkeley-hardfloat/        # Vendored HardFloat
MXP_Tools/                             # Python verification toolkit (gen/emit/ref/compare/viz)
docs/                                  # specs, plans, kickoff notes
```

### Source provenance

`Accumulator*.v`, `PE_*.v`, `SystolicArray.v`, etc. are **imported** from the sibling
MXP project; `sram_1rw*.v` from the sibling SRAM project. Both upstreams are the source
of truth — fix algorithm/microarchitecture issues upstream and re-import rather than
patching the local copies. `GEMM.v` and `RMW.v` are project-local glue and edited freely.

## Verification toolkit (`MXP_Tools/`)

Fork of the upstream `MXP_Tools` with project-specific additions (`rmw_gen.py`,
`rmw-gen` CLI, 16/32-bank interleaved mappings). Generates HW inputs and the SW golden
GEMM, and compares HW SRAM dumps (one `$writememh` file per bank) against the golden.

```bash
cd MXP_Tools
python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/A8_B8
python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8
python -m mxp_tools compare --ref ... --hw-banks ... --layout interleaved_row_major_32bank
```

> **Naming gotcha:** `--prec-a` = WEIGHT precision (our `B_PREC`), `--prec-b` =
> ACTIVATION precision (our `A_PREC`). Symmetric modes hide a swap; asymmetric modes
> catch it as all-zero dumps.

## License

See upstream MXP / SRAM projects for the imported RTL terms.
