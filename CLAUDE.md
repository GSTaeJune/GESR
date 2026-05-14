# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Next session kickoff (2026-05-14, RMW unit done — system integration up next)

RMW unit (Tasks 1–9, see git history / `docs/superpowers/`) is **complete and standalone-verified** (`bash sim/run_rmw.sh` → 71/71 PASS). Next phase: **GEMM ↔ RMW ↔ sram_1rw_banked 시스템 통합**.

다음 세션 시작 protocol — 사용자가 "통합 설계 시작" / "다음 단계 시작" / "brainstorming 시작" 등으로 시작하면:

1. **`docs/next-session-kickoff.md` 를 먼저 읽는다** — 통합 단계의 4가지 열린 설계 질문과 브레인스토밍 시작 컨텍스트가 정리돼 있음.
2. `superpowers:brainstorming` 스킬을 호출해서 그 4개 질문(아래 "What's NOT settled" 섹션 항목 2/3/4/6)을 같이 정리한다. 결정된 답들을 `docs/superpowers/specs/2026-MM-DD-integration-design.md`로 저장.
3. spec이 끝나면 `superpowers:writing-plans` → `docs/superpowers/plans/2026-MM-DD-integration-implementation.md`.
4. plan이 끝나면 implementation 진행.

이전 단계(RMW unit)가 어떻게 굴러갔는지 보고 싶으면 `docs/superpowers/specs/2026-05-14-rmw-design.md` + `docs/superpowers/plans/2026-05-14-rmw-implementation.md` 참고. 같은 흐름을 답습하면 됨.

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

## Top-level wiring (current state vs target)

Top module is `GEMM` (set in `gemm_sram.xpr`). At present `GEMM.v` is the unmodified MXP `TOP` — it exposes `out_accumulate[32*60-1:0]` / `out_scale[32*4*9-1:0]` / `out_fire[31:0]` to module ports. SRAM is sitting in the project but is **not yet instantiated** anywhere. `RMW.v` is the standalone FP32 RMW unit (`int_to_fp32` + `L_CONV`-cycle SRAM delay + `fp32_adder`, total latency `L_CONV + L_ADD = 5`); it's verified standalone (71/71 vectors via `sim/run_rmw.sh`) but **not yet wired** between GEMM and the SRAM banks.

Target end-state, conceptually:

```
GEMM (INT psum + 9-bit scale) ──► RMW (INT→FP32, FP32 add) ──► sram_1rw_banked (FP32)
                                       ▲                              │
                                       └────── FP32 prior psum ◄──────┘
```

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

### Open questions specific to RMW

- **Granularity**: one RMW per column (32 instances) or per lane (128 instances)? `in_GEMM[31:0]` is a single 32-bit word, but `out_accumulate` is 60 bits per column packed as A8/A4/A2 layouts (see "MXP control surface" below). For A4 (two 18-bit) and A2 (four 15-bit), one column produces multiple independent psums each with its own scale sub-word → likely per-lane RMW with mode-aware muxing on the input side. For A8 there is only one effective lane, so 3 of 4 RMWs idle.
- **Latency budget**: INT→FP32 dequant + FP32 add is multi-cycle (DSP48-based FP add IP is typically 3–6 cycles). Decide whether RMW is pipelined or single-cycle multi-stage; the controller's read-issue timing depends on this.
- **FP32 adder implementation**: Vivado Floating-Point IP vs hand-written unpack/align/add/normalize/repack. The skeleton doesn't commit.
- **Saturation / NaN handling**: behavior when `in_SRAM` is uninitialized (NaN bit pattern). The first K-tile RMW must either read a guaranteed-zero FP32 (`32'h00000000`) or be gated to skip the read and write the dequant result directly.

Bank addressing is also open: `sram_1rw_banked` defaults to `NUM_BANKS=16, BANK_DEPTH=32768` (2 MB total). Map 32 GEMM columns onto banks. Note `BANK_STRATEGY="INTERLEAVED"` (LSB bank-select) vs `"SEQUENTIAL"` (MSB bank-select) — see `../sram/docs/banking-wrapper-decisions.md`.

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

1. **Vivado GUI** — open `gemm_sram.xpr`, set the testbench as simulation top, Run Behavioral Simulation. Top synth module is `GEMM`.
2. **XSim batch** — `sim/run_*.sh` scripts already exist for the RMW unit and its sub-modules:

   ```bash
   bash sim/run_rmw_smoke.sh      # HardFloat round-trip smoke (no DUT logic)
   bash sim/run_int_to_fp32.sh    # int_to_fp32 unit TB (6 directed cases)
   bash sim/run_fp32_adder.sh     # fp32_adder unit TB (4 directed cases)
   bash sim/run_rmw.sh            # full RMW vector TB (71 cases — see below)
   ```

   **Full RMW vector test** requires `MXP_Tools` to generate the vector files first:

   ```bash
   cd MXP_Tools && python -m mxp_tools rmw-gen --out work/rmw --n 64 --seed 0 && cd ..
   bash sim/run_rmw.sh             # expect: rmw_tb: ALL 71 TESTS PASSED
   ```

   To author a new TB, copy the pattern from any existing `sim/run_*.sh`. Required directive that **must** appear in both RTL and TB or XSim errors out: `` `timescale 1ns/1ps ``.

When parameter overrides are needed at sim time, **do not use `xelab -generic_top "P=V"`** — there's a cmd.exe quoting bug under Git Bash on Windows that strips the `=`. Use `` `ifdef `` in the source + `xvlog -d FLAG` instead. The SRAM TBs already use this pattern:

- Leaf TB (`sram_1rw_tb.v`): `USE_PIPELINE`, `RUN_INIT_TEST`.
- Banked TB (`sram_1rw_banked_tb.v`): `USE_PIPELINE`, `USE_SEQUENTIAL` (selects `BANK_STRATEGY="SEQUENTIAL"` vs default `"INTERLEAVED"`).

**TB sizing convention.** Use a small config for fast sim (banked TB uses 4 banks × 64 depth × 32-bit), full-size parameters only via the elab sweep (`../sram/sim/run_wrapper_elab.sh`). When writing this project's TBs, follow the same pattern — do not run a 2 MB sim by default.

## Verification helper: `../MXP_Tools/` (Python)

Generates HW inputs and SW golden GEMM. From `../MXP_Tools/README.md`:

```bash
cd ../MXP_Tools
python -m mxp_tools gen     --out work/ -M 128 -K 128 -N 128 --seed 0
python -m mxp_tools emit    --out work/                    # all 9 precision modes
python -m mxp_tools ref     --out work/                    # SW golden
# run HW → produces work/hw_out/bank*.mem ($writememh)
python -m mxp_tools compare --ref work/sw_ref/C_sw_mxint8_mxint8.npz \
                            --hw-banks work/hw_out/bank0.mem \
                            --layout single
```

For this project the HW dump is one `$writememh` file **per SRAM bank**. The Python side's bank → `(m, n)` mapping is plug-in (`hwio.gather_banks(..., mapping)`); the mapping for our RMW layout must match what the testbench dumps. Once you finalize bank-column mapping, write a custom mapping callable rather than relying on `default_single_bank_row_major` / `default_banks_split_rows`. Numerical contract: HW output words are interpreted as IEEE-754 FP32 bit patterns (so the RMW path must dequantize before writing).

## What's NOT settled (open design questions)

These are explicit unknowns — do not invent answers without asking. They are the work this project exists to do. (Items previously open but **now resolved** by the `RMW.v` contract are noted below.)

1. ~~**GEMM-output → SRAM word mapping.**~~ **Resolved**: dequantize to FP32 at RMW, store FP32 in SRAM, 32-bit `DATA_WIDTH`.
2. **Bank-to-column assignment.** 32 columns into N banks. Interleaved vs sequential. Per-column FIFO vs 32 parallel RMW engines vs serialized arbiter.
3. **Fire-timing aware scheduling.** Per-column `out_fire` skew across the chain — the RMW controller must not collide two columns onto the same bank in the same cycle. Made harder by RMW's multi-cycle FP32 add.
4. **First-tile init.** First RMW for a given output tile must not read garbage from uninitialized FP32 SRAM. Either zero-prime the SRAM image via a `do_write` loop at the start of sim (banked wrapper has no `INIT_FILE`), or gate the first read out and write the dequant result directly.
5. ~~**Scale storage.**~~ **Resolved**: scale is consumed during INT→FP32 conversion in RMW, not stored.
6. **RMW granularity (new).** Per-column (32 instances) vs per-lane (128 instances) — driven by A8/A4/A2 multi-lane packing.
7. **RMW latency / impl (new).** FP IP vs hand-written; pipelined latency budget; how the read-issue cadence aligns with it.

When implementing, write the decision into the `modify.v` header and (if architectural) into a short doc under a future `docs/` folder. Mirror the `../sram/docs/` pattern.

## File layout

```
gemm_sram.xpr                          # Vivado 2024.1 project (target xc7vx485tffg1157-1)
gemm_sram.srcs/sources_1/
    new/
        GEMM.v                         # Top — currently the unmodified MXP TOP
        RMW.v                          # FP32 RMW unit (int_to_fp32 + delay + fp32_adder)
        int_to_fp32.v                  # INT32 + 9-bit signed scale -> IEEE-754 FP32
        fp32_adder.v                   # IEEE-754 FP32 adder (HardFloat-based)
    imports/Desktop/MXP/...            # MXP compute RTL (mirrored from ../MXP/);
                                       # Accumulator_Col.v has IMPLICIT_total patch
    imports/Desktop/sram/rtl/...       # SRAM RTL (mirrored from ../sram/)
tb/                                    # Unit testbenches (rmw_tb, int_to_fp32_tb,
                                       # fp32_adder_tb, rmw_smoke_tb, accumulator_col_elab)
sim/                                   # XSim batch run scripts (run_*.sh + clean.sh)
third_party/berkeley-hardfloat/        # Vendored HardFloat Verilog (Chisel→Verilog
                                       # via sbt). One file: HardFloatBundle.v
                                       # See VENDORING.md for module names + ports.
docs/superpowers/                      # Specs + plans for completed RMW unit work
docs/hardfloat-setup.md                # HardFloat re-vendoring guide (sbt + Chisel)
gemm_sram.{sim,cache,hw,ip_user_files} # Vivado scratch — generated, do not edit
```

Adjacent projects (used as references, do not edit from here):
```
../MXP/         — MXP standalone (250 MHz wrapper closure, area/power numbers)
../sram/        — SRAM standalone (leaf + banked, both with full TB)
../MXP_Tools/   — Python verification toolkit (golden GEMM + 3-way diff + viz)
```
