# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Next session kickoff (2026-05-15, integration done — synthesis / future scope 단계로)

GEMM ↔ RMW ↔ sram_1rw_banked **시스템 통합 완성 및 검증**됨 — `bash sim/run_integration_sweep.sh` → `ALL 9 MODES PASSED` (9 precision combinations A,B ∈ {2,4,8}, 각각 128×128 = 16384 element 모두 bit-exact 일치 vs MXP_Tools golden). 단위 검증도 그대로 유효: `bash sim/run_rmw.sh` → 71/71 PASS.

다음 세션 시작 protocol — `docs/next-session-kickoff.md` 를 먼저 읽고, 사용자가 어떤 방향으로 가고 싶은지에 따라:

1. **잠정값 재검토** (throughput / area) — 통합 spec § 9 의 잠정 결정값 표 (RMW instance 수 1개 / loop order K-outermost / 워크로드 128³ / INTERLEAVED bank) 의 재검토 트리거가 발생한 시점. 새 spec 필요.
2. **Timing closure / 합성** — Vivado 합성, 250 MHz wrapper / xc7vx485 target. MXP standalone 의 closure 패턴 답습.
3. **Future scope: SRAM weight storage** — 통합 spec § 1 "Future scope" 항목. 별도 spec 필요.
4. **9-mode 자동 회귀 CI** — `sim/run_integration_sweep.sh` 를 CI 로 묶기.

이전 단계 흐름:
- RMW unit: `docs/superpowers/specs/2026-05-14-rmw-design.md` + `docs/superpowers/plans/2026-05-14-rmw-implementation.md`
- 통합: `docs/superpowers/specs/2026-05-14-integration-design.md` + `docs/superpowers/plans/2026-05-14-integration-implementation.md`

---

## Project Goal

`gemm_sram` = **integration** of the MXP mixed-precision bit-serial systolic array (GEMM compute) with the parameterized 1RW SRAM banks. The end state is a **Read-Modify-Write (RMW)** datapath: a column's accumulated GEMM result is added to the prior partial sum held in SRAM, then written back to the same address so the SRAM stores the running tile-wise sum.

This is a Vivado 2024.1 project (`gemm_sram.xpr`, target `xc7vx485tffg1157-1`) that **imports** RTL from two sibling sister projects rather than maintaining its own copies. Both upstream repos are the source of truth — do not mutate imported sources without understanding the upstream contract.

## Source provenance (do not edit imports in place)

| Module | Imported from | Role |
|---|---|---|
| `GEMM.v` (top), `RMW.v` | `gemm_sram.srcs/sources_1/new/` | **Project-local** — RMW glue lives here. Edit freely. |
| `Accumulator.v`, `Accumulator_Col.v`, `PE_feeder.v`, `PE_naive.v`, `SystolicArray.v`, `adder_lane.v`, `station.v` | `../../MXP/MXP.srcs/sources_1/new/` | MXP compute engine (32×32 bit-serial systolic). |
| `sram_1rw.v`, `sram_1rw_banked.v` | `../../sram/rtl/` | Parameterized 1RW SRAM (leaf + banked wrapper). |

Vivado's import keeps copies under `gemm_sram.srcs/sources_1/imports/Desktop/{MXP,sram}/...`. If you change algorithm/microarchitecture, fix it upstream (`../MXP/` or `../sram/`) and re-import; do not patch the local copy and forget. Refer to `../MXP/CLAUDE.md` and `../sram/CLAUDE.md` for upstream conventions and gotchas (Verilog-2001 only in SRAM, `in_a`=weight/`in_b`=activation naming gotcha in MXP, etc.).

## Top-level wiring

Top module is `gemm_sram_top` (set in `gemm_sram.xpr`; sim target is `gemm_sram_top_tb`). The wrapper is **pure structural** — GEMM + RMW + sram_1rw_banked instantiated with just two glue lines (Q→RMW.in_SRAM hardwire, sram_D_use_zero mux between zero and RMW.out_RMW). All controller responsibility lives in the TB (`tb/gemm_sram_top_tb.v`) — no synthesizable FSM in the wrapper.

Datapath:

```
GEMM (INT psum + 9-bit scale) ──► RMW (INT→FP32, FP32 add) ──► sram_1rw_banked (FP32)
                                       ▲                              │
                                       └────── FP32 prior psum ◄──────┘
```

Verified end-to-end via `bash sim/run_integration_sweep.sh` (9/9 modes bit-exact PASS vs MXP_Tools golden, 128×128 = 16384 elements per mode).

### RMW contract (from `RMW.v` header — committed decisions)

```verilog
module RMW (
    input  wire        clk, rst,
    input  wire [31:0] in_SRAM,   // FP32 prior partial sum read from SRAM
    input  wire [31:0] in_GEMM,   // INT32 from GEMM accumulator (one lane's worth)
    input  wire [8:0]  scale,     // combined 9-bit scale (matches one sub-word
                                  // of Accumulator_Col `out_scale`, = scale_len+1)
    output reg  [31:0] out_RMW    // FP32 sum to write back to SRAM
);
```

This pins down **two** of the previously-open questions:

1. **Bit-width reconciliation**: dequantize INT→**FP32 at the RMW boundary**. Both SRAM storage and the FP32 add live in IEEE-754 single precision — 32-bit words, no `DATA_WIDTH` widening, no INT-domain accumulation in SRAM. This also lines up with `MXP_Tools/hwio.py::read_writememh_fp32` which already assumes FP32 bit patterns in the `$writememh` dump.
2. **Scale handling**: the 9-bit `scale` is the per-sub-word combined `(act_scale + weight_scale − 127)` that `Accumulator_Col` produces (note: 9-bit signed, `{1'b0, act_scale} + {1'b0, weight_scale} - 9'sd127`). RMW consumes it during the INT→FP32 conversion; it is **not** stored alongside the psum.

The RMW controller (whether inside RMW or wrapping it) must still:
1. On `out_fire[col]` rising, latch the lane's INT32 psum and the corresponding 9-bit scale sub-word.
2. Issue a READ to the SRAM address holding that lane's prior FP32 psum.
3. Wait for the read return (1 cycle if leaf `PIPELINE=0`, 2 cycles if `PIPELINE=1`).
4. Convert INT→FP32 (using `scale`), add to `in_SRAM`, write back.

### Resolved (during integration) — committed values

- **Granularity**: **1 RMW instance** (jointly serves all 32 cols / all lanes via TB-side mode-aware dispatch). Lane decode (A8 = 1, A4 = 2, A2 = 4 RMW dispatches per col fire) lives in `tb/gemm_sram_top_tb.v` capture-block per `docs/superpowers/notes/lane-to-c-mapping.md`. Re-visit trigger: throughput / area / sim-time bottleneck.
- **Latency budget**: `L_CONV = 2`, `L_ADD = 3`, total = **5 cycles**. Hand-written via vendored HardFloat (`third_party/berkeley-hardfloat/`).
- **FP32 adder implementation**: **HardFloat-based** (`fp32_adder.v` wraps `addRecFN`). Not Vivado FP IP.
- **First-tile init**: **SRAM zero-prime via TB loop** at sim start (16384 words written through the `sram_D_use_zero=1` mux), then `sram_D_use_zero=0` for the rest. No NaN risk.
- **Bank addressing**: `NUM_BANKS = 16`, `BANK_DEPTH = 32768`, `BANK_STRATEGY = "INTERLEAVED"`, `PIPELINE = 0`. `C[m,n] → flat=m*N+n → bank=flat%16, word=flat//16`. Mapping callable in `MXP_Tools/mxp_tools/hwio.py::interleaved_row_major_16bank`. 16384 / 16 = 1024 words/bank for the 128×128 workload.

## MXP control surface (you will need this)

`GEMM` is driven by two distinct chains — getting these wrong silently produces dead lanes. From `../MXP/CLAUDE.md`:

- **Station data chain**: enters at col 31, propagates **leftward** (col 31 → col 0). Carries `in_Scale_Activation`, `in_Station_control` (precision: A8/A4/A2/idle), `in_loadEN`. Aligned with `in_b` (activation) chain.
- **Station selector chain** (`in_station_control`): enters at col `num_col/2=16`, fans **outward symmetrically**. Toggles Buf1↔Buf2 ping-pong. Same topology as the Accumulator chain so the toggle wave aligns with the fire window.
- **Accumulator chain**: `in_start_accumulate`, `in_Wcontrol`, `in_scale_weight` enter at col 16, fan outward.

`out_fire` is **per-column**, not global — column fire timings are offset by the chain delays. Any SRAM scheduler must honor this, e.g., a per-column FIFO or 32 parallel RMW engines.

Per-column lane slicing of `sa_out` (in GEMM.v at `col_slice`): MSB-first lane order `{psum0, psum1, psum2, psum3}` — `psum0` carries the top weight-bit pair, `psum3` the bottom. The downstream `Accumulator_Col` combines these via a `<<2/<<4` adder tree depending on precision mode (`Mode_oh` one-hot {A2,A4,A8}).

Precision mode → `out_accumulate` layout (from `Accumulator_Col.v`):
- A8: single 21-bit signed value, sign-extended into 60 bits.
- A4: two 18-bit signed values `{0..0, s1_a, s1_b}` packed low.
- A2: four 15-bit raw lane accumulators `{out_INT2_0..3}` packed.

The RMW datapath must respect this layout when adding the prior SRAM psum.

## SRAM control surface

From `../sram/CLAUDE.md` — **active-low** `CEB` (chip enable bar) and `WEB` (write enable bar), **active-high** `WMASK` (bit-write enable, 1 = write). Read latency is 1 cycle when `PIPELINE=0`, 2 cycles when `PIPELINE=1`. The leaf bank has no reset on `mem[]` — power-up state is whatever simulation initialized to. **`INIT_FILE` exists only on the leaf**; the banked wrapper has no `INIT_FILE` parameter (deliberate — Q5 in `../sram/docs/banking-wrapper-decisions.md`). For tests, initialize via a `do_write` loop instead of file preload.

The banked wrapper additionally has a 1-cycle `bank_sel_d1` register inside, so its effective read latency is unchanged from the leaf's `PIPELINE` setting — but mux-select alignment must use `bank_sel_d1`, not raw `bank_sel`. Important if you ever bypass the wrapper. The wrapper's output mux is `wire [DW-1:0] bank_q [0:NB-1]` with indexed access (`bank_q[bank_sel_d1]`) — clean parametric pattern; reuse it if you write a custom mux.

**Polarity inversion at PNR.** Foundry macros usually expose `BWEB` (active-low bit-write) instead of our active-high `WMASK`. `../sram/rtl/sram_1rw_macro_wrap.v.example` is the template — `assign BWEB = ~WMASK;` plus a vendor macro instance. When this project moves to PNR/silicon, copy that template and swap the leaf instance; wrapper RTL stays unchanged.

## Common commands

Two ways to run sim:

1. **Vivado GUI** — open `gemm_sram.xpr`. Synth top = `gemm_sram_top`; sim top = `gemm_sram_top_tb`. Run Behavioral Simulation.
2. **XSim batch** — `sim/run_*.sh` scripts:

   ```bash
   # Unit-level
   bash sim/run_rmw_smoke.sh         # HardFloat round-trip smoke (no DUT logic)
   bash sim/run_int_to_fp32.sh       # int_to_fp32 unit TB (6 directed cases)
   bash sim/run_fp32_adder.sh        # fp32_adder unit TB (4 directed cases)
   bash sim/run_rmw.sh               # full RMW vector TB (71 cases)
   bash sim/run_top_elab.sh          # gemm_sram_top elab-only smoke
   bash sim/run_integration_smoke.sh # TB zero-prime + dump (no GEMM driving)

   # Integration (end-to-end vs MXP_Tools golden)
   bash sim/run_integration_one.sh <LABEL> <A_PREC> <B_PREC>   # 1 mode (e.g. "A8_B8" 8 8)
   bash sim/run_integration_sweep.sh                            # all 9 modes serial
   bash sim/run_integration_parallel.sh                         # print parallel dispatch guide
   ```

   **Full RMW vector test** requires `MXP_Tools` to generate the vector files first:

   ```bash
   cd MXP_Tools && python -m mxp_tools rmw-gen --out work/rmw --n 64 --seed 0 && cd ..
   bash sim/run_rmw.sh             # expect: rmw_tb: ALL 71 TESTS PASSED
   ```

   **Integration sweep** is self-contained — invokes `gen / emit / ref` internally then `run_integration_one.sh` + `compare`. Last line on success: `ALL 9 MODES PASSED`. Runtime ~20–25 min for the full 9-mode sweep.

   To author a new TB, copy the pattern from any existing `sim/run_*.sh`. Required directive that **must** appear in both RTL and TB or XSim errors out: `` `timescale 1ns/1ps ``.

   **Gotcha — `-testplusarg "K=V"` under Git Bash**: the `=` is stripped before xsim sees it. Wrap with `cmd //c "xsim ... -testplusarg \"K=V\""` — the integration scripts already do this.

When parameter overrides are needed at sim time, **do not use `xelab -generic_top "P=V"`** — there's a cmd.exe quoting bug under Git Bash on Windows that strips the `=`. Use `` `ifdef `` in the source + `xvlog -d FLAG` instead. The SRAM TBs already use this pattern:

- Leaf TB (`sram_1rw_tb.v`): `USE_PIPELINE`, `RUN_INIT_TEST`.
- Banked TB (`sram_1rw_banked_tb.v`): `USE_PIPELINE`, `USE_SEQUENTIAL` (selects `BANK_STRATEGY="SEQUENTIAL"` vs default `"INTERLEAVED"`).

**TB sizing convention.** Use a small config for fast sim (banked TB uses 4 banks × 64 depth × 32-bit), full-size parameters only via the elab sweep (`../sram/sim/run_wrapper_elab.sh`). When writing this project's TBs, follow the same pattern — do not run a 2 MB sim by default.

## Verification helper: `MXP_Tools/` (Python, local subdir)

Generates HW inputs and SW golden GEMM. Typical invocation (the sweep script already wraps these):

```bash
cd MXP_Tools
python -m mxp_tools gen   --out ../work/A8_B8 -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit  --out ../work/A8_B8                        # emits all 3 precs
python -m mxp_tools ref   --out ../work/A8_B8 --prec-a 8 --prec-b 8   # SW golden
# run HW → produces work/A8_B8/hw_out/bank{0..15}.mem ($writememh)
BANKS=$(printf "../work/A8_B8/hw_out/bank%d.mem " {0..15})
python -m mxp_tools compare --ref ../work/A8_B8/sw_ref/C_sw_mxint8_mxint8.npz \
                            --hw-banks ${BANKS} \
                            --layout interleaved_row_major_16bank
```

**Naming gotcha (was a Task 8 silent-bug source)**: MXP_Tools' `--prec-a` = WEIGHT precision = our plusarg `B_PREC` (weight is the bit-serial first operand in `mxint_gemm_golden(A, ..., prec_A=weight)`). `--prec-b` = ACTIVATION precision = our `A_PREC`. The resulting `.npz` filename slot order is `C_sw_mxint{prec_a}_mxint{prec_b}.npz` = `C_sw_mxint{B_PREC}_mxint{A_PREC}.npz`. Symmetric modes (A2_B2, A4_B4, A8_B8) hide the bug; asymmetric modes catch it as all-zero HW dumps. The sweep + per-mode scripts already use the corrected convention.

HW dump = one `$writememh` file per SRAM bank (16 files). Mapping callable: `MXP_Tools/mxp_tools/hwio.py::interleaved_row_major_16bank`. Numerical contract: HW output words are interpreted as IEEE-754 FP32 bit patterns (RMW dequantizes INT→FP32 before storing).

## Settled — with re-visit triggers

All design questions from the original spec § 9 are now committed with concrete values (see "Resolved" subsection above + the integration spec at `docs/superpowers/specs/2026-05-14-integration-design.md` § 9). Trigger list for re-opening any of them:

| Decision | Committed value | Re-visit trigger |
|---|---|---|
| RMW instance count | 1 (TB-side mode-aware dispatch) | throughput / area goal; sim time too long |
| Loop order | K-outermost | dataflow re-optimization phase |
| Workload (M, K, N) | (128, 128, 128) | sim time > 10 min |
| Bank strategy | INTERLEAVED | bank-conflict measurement; SEQUENTIAL comparison needed |
| SRAM PIPELINE | 0 | timing closure phase |
| RMW (L_CONV, L_ADD) | (2, 3), total 5 cyc | timing closure phase |
| First-tile init | TB zero-prime via mux | future hardware controller |
| Bank-to-column mapping | row-major flat → bank=flat%16, word=flat//16 | non-128² workload |

Future scope (not in current spec): use SRAM also as **B/weight input storage** (currently TB drives `in_b` directly via `$readmemh`). Separate spec needed.

## File layout

```
gemm_sram.xpr                          # Vivado 2024.1 project (target xc7vx485tffg1157-1)
                                       # synth top = gemm_sram_top; sim top = gemm_sram_top_tb
gemm_sram.srcs/sources_1/
    new/
        gemm_sram_top.v                # Integration wrapper (GEMM + RMW + SRAM, pure structural)
        GEMM.v                         # MXP TOP wrapper (instantiated inside gemm_sram_top)
        RMW.v                          # FP32 RMW unit (int_to_fp32 + delay + fp32_adder)
        int_to_fp32.v                  # INT32 + 9-bit signed scale -> IEEE-754 FP32
                                       # (Task 7 fix: zero-passthrough for in_int=0 + scale≠127)
        fp32_adder.v                   # IEEE-754 FP32 adder (HardFloat-based)
    imports/Desktop/MXP/...            # MXP compute RTL (Accumulator_Col.v has IMPLICIT_total patch)
    imports/Desktop/sram/rtl/...       # SRAM RTL (sram_1rw + sram_1rw_banked)
tb/
    gemm_sram_top_tb.v                 # Integration TB (9-mode plusarg-driven)
    rmw_tb.v, int_to_fp32_tb.v,
    fp32_adder_tb.v, rmw_smoke_tb.v,
    accumulator_col_elab.v             # Unit testbenches
sim/
    run_top_elab.sh                    # gemm_sram_top elab smoke
    run_integration_smoke.sh           # TB zero-prime + dump (no GEMM driving)
    run_integration_one.sh             # 1 mode end-to-end (<LABEL> <A_PREC> <B_PREC>)
    run_integration_sweep.sh           # 9-mode serial sweep + compare gate
    run_integration_parallel.sh        # 9-mode parallel dispatch guide (print template)
    run_rmw*.sh, run_int_to_fp32.sh, run_fp32_adder.sh, clean.sh
third_party/berkeley-hardfloat/        # Vendored HardFloat (HardFloatBundle.v + VENDORING.md)
MXP_Tools/                             # Python verification toolkit (golden GEMM + compare + viz)
    mxp_tools/                         # gen / emit / ref / compare / viz / rmw-gen subcommands
    tests/test_hwio_interleaved.py     # pytest for interleaved_row_major_16bank mapping
docs/
    next-session-kickoff.md            # Phase-handoff doc (updated each major phase end)
    hardfloat-setup.md                 # HardFloat re-vendoring guide (sbt + Chisel)
    superpowers/
        specs/2026-05-14-rmw-design.md
        specs/2026-05-14-integration-design.md
        plans/2026-05-14-rmw-implementation.md
        plans/2026-05-14-integration-implementation.md
        notes/mxp-driving-sequence.md         # Task 5 — gemm_sram TB perspective on driving
        notes/lane-to-c-mapping.md            # Task 6 — A4/A2 lane → C[m,n] dispatch table
precision_modes_protocol.md            # MXP driving protocol (v1.0, validated 2026-05-11)
gemm_sram.{sim,cache,hw,ip_user_files} # Vivado scratch — generated, do not edit
work/<LABEL>/                          # Integration sim work dirs (gitignored, generated by sweep)
```
