# BF16 Native Datapath Implementation Plan (A6)

> **For agentic workers:** This plan is executed inline by its author in the same session
> (design + code by one context, then adversarial review per project rule 2). Algorithm
> spec below is normative; the RTL files carry the same content as heavily-commented
> Verilog — the user will read the RTL directly, so the *code comments* are a first-class
> deliverable (Korean, per project convention).

**Goal:** Replace the fp32-domain HardFloat detour in the RMW bf16 datapath with
hand-written native bf16 arithmetic — `bf16_adder` and `int_to_bf16` re-implemented at
true bf16 width (8-bit significand internal datapaths), removing HardFloat from the RMW
path entirely.

**Why (user decision 2026-07-13):** Phase 2c honest timing showed the bottleneck is the
monolithic fp32 `AddRecFN` stage (est. 12.4 ns pre-place) — a 24-bit-significand adder
doing 8-bit work. A right-sized bf16 adder is ~`fp32_to_bf16_rne`-scale logic per stage
(alignment <= ~11 positions, 12-bit add, 11-bit LZC), so each of the existing 5 pipeline
stages becomes trivially fast for 250 MHz PNR, and area drops. This is finding A6
(reduced-significand adder) from the 2026-07-12 RTL analysis, upgraded from
BIG-BUT-RISKY to the main line by user directive: "다시 BF16 계산기로 RTL 짜".

**Architecture:** Same module interfaces, same parameter surfaces, same RMW 5-cycle
external contract, same oracle gates. Only the *internals* of `bf16_adder.v` and
`int_to_bf16.v` change (single-round native RNE arithmetic). All existing bit-exact
gates arbitrate correctness; oracle vectors are *strengthened* (never weakened) with
directed edges for newly hand-written paths.

**Tech stack:** Verilog-2001, XSim (bash sim/run_*.sh), ml_dtypes oracle
(sim/bf16_vectors.py), Vivado OOC synth for relative timing/area proxy.

---

## 1. Frozen contracts (MUST NOT change)

| Surface | Contract |
|---|---|
| `bf16_adder` ports/params | `#(L_IN=0, L_ADD=3, L_SUM=0, L_OUT=0)`, ports `clk,rst,a[15:0],b[15:0],sum[15:0]`. Total latency = L_IN+L_ADD+L_SUM+L_OUT registers. |
| `int_to_bf16` ports/params | `#(L_CONV=2)`, ports `clk,rst,in_int[31:0],scale[8:0],out_bf16[15:0]`. Total latency = L_CONV registers (>=1). |
| `RMW` | Unchanged instantiation: `int_to_bf16 #(L_CONV-1)` + `sram_dly[0:L_CONV-2]` + `bf16_adder #(1, L_ADD-2, 1, 1)`. Total = L_CONV+L_ADD = 5 cy. `out_RMW` registered output stays. |
| Numeric truth | `sim/bf16_vectors.py` (ml_dtypes) for units; `rmw_gen.py` for RMW; MXP_Tools golden `--accum bf16` for integration. Gates: int_to_bf16 / bf16_adder / rmw 113 / 9-mode / mixed 3 — all bit-exact. If a gate goes red: fix RTL, never weaken vectors. |
| Skew gate | `rmw_tb` streaming (1 vec/cy) is the SOLE latency/skew gate. Do not weaken. |

Numeric semantics being implemented (from the oracle, unchanged):

- `bf16_adder`: `sum = bf16_RNE(a + b)` — true IEEE bfloat16 addition (1+8+7, RNE,
  subnormals, +-inf, NaN). Oracle computes `(f32(a)+f32(b)).astype(bfloat16)` which
  equals single-round bf16 add by the 2p+2 double-rounding theorem (24 >= 2*8+2).
  NaN result: TB is sign/payload-agnostic; RTL emits canonical `0x7FC0`.
- `int_to_bf16`: `out = bf16_RNE( r8 * 2^(sc - 127) )` where `r8 = int32 -> bf16 RNE`
  (first rounding, 8-bit significand) and `sc` = 9-bit **signed** scale. Includes:
  zero passthrough (+0), deep-underflow flush to signed zero, subnormal band with
  second RNE (denormalization rounding — the exact defect class that killed v1),
  inf saturation for e' >= 255.

## 2. Native `bf16_adder` algorithm (single-round RNE, 3 comb blocks)

Operand fields: `s`, `e[7:0]`, `f[6:0]`. `sig8 = {e!=0, f}` (hidden bit; subnormal
gets 0), effective biased exponent `ee = (e==0) ? 1 : e` (subnormals live at the
min-normal binade with hidden 0 — this makes alignment uniform).

**Block A — unpack / specials / swap / align / add-sub:**
1. Specials: `a_nan|b_nan -> NaN`; `a_inf & b_inf & (sa^sb) -> NaN`; else any inf
   -> that inf. Flags ride the pipeline; general path result is overridden at pack.
2. Magnitude compare on `{e,f}` (15b unsigned — bf16 ordering property). Bigger -> `big`,
   other -> `small`; ties pick `a` (either is correct: equal magnitudes are exact cases).
3. `d = ee_big - ee_small`. Align small into an 11-bit grid `{sig8, G, R, S}` =
   `{sig8_small, 3'b000} >> d`, all dropped bits OR into the S (bit0) position; for
   d >= 11 small collapses to pure sticky. Implemented as a 22-bit funnel:
   `wide = {body11, 11'b0} >> min(d, 11)`, `small_op = {wide[21:12], wide[11] | |wide[10:0]|}`.
4. Effective op `eff_sub = sa ^ sb`. Magnitude arithmetic in 12 bits:
   `m12 = eff_sub ? big_op - small_op : big_op + small_op`
   (subtraction is non-negative by the swap; sticky-OR-into-bit0 *before* subtraction
   is the standard GRS scheme — the inexactness marker survives as bit0 of the result).

**Block B — normalize:**
- Add-carry (`m12[11]`, effective-add only): right shift 1 with sticky merge
  (`{m12[11:2], m12[1]|m12[0]}`), `e' = ee_big + 1`.
- Else LZC over `m12[10:0]`, left shift `sh = min(lzc, ee_big-1)` (exponent floored at
  biased 1 = subnormal clamp), `e' = ee_big - sh`.
  Invariant (classic): massive cancellation (lzc >= 2) only occurs for d <= 1, where
  alignment dropped no bits (G=R=S=0, exact) — so left-shifting the GRS grid never
  shifts a sticky bit into a numeric position. For d >= 2 the result MSB is at bit10
  or bit9 (shift <= 1) and GRS stay meaningful.
- `m12 == 0` -> exact zero (see sign rule below). Adder zero results are always exact
  (subnormal grid closed under +-; normal cancellations >= 2^-133), so no
  "inexact rounds to zero" case exists.

**Block C — round / pack:**
- RNE on the 11-bit grid: `up = G & (L | R | S)` = `m[2] & (m[3] | m[1] | m[0])`.
- `sig9 = m[10:3] + up`; carry (`0xFF -> 0x100`) renormalizes: `sig8 = 0x80, e'+1`.
- Encode: `e'' >= 255 -> {sign, FF, 0}` (RNE overflow -> inf; whenever e'' hits 255 the
  true value >= (2-2^-8)*2^127 so inf is the correct rounding);
  `sig8 == 0 -> signed zero`; `sig8[7] == 0 -> subnormal {sign, 8'd0, sig8[6:0]}`
  (e' is 1 by the clamp); else normal `{sign, e''[7:0], sig8[6:0]}`.
- Zero sign (IEEE RNE): exact-zero result -> `+0` for effective subtraction,
  `sign_big` for effective addition (covers +0+ +0=+0, -0+ -0=-0, +0+ -0=+0, x-x=+0).
- Specials mux last: NaN -> `0x7FC0`, inf -> `{inf_sign, FF, 0}`.

**Pipeline cuts (params = shift-reg depth at a named cut):**
`L_IN` on raw a/b -> [Block A] -> `L_ADD` on {m12, ee_big, sign, eff_sub, specials}
-> [Block B] -> `L_SUM` on {m_norm, e_norm, flags} -> [Block C] -> `L_OUT` on bf16.
Defaults (0,3,0,0) keep total latency 3 = old default; RMW's (1,1,1,1) gives 4 tiny
stages. Function is register-placement-invariant; dual-DUT TB covers both configs.

## 3. Native `int_to_bf16` algorithm (two comb halves)

**Front half — int32 -> r8 (first RNE):**
1. `sign = in_int[31]`, `mag = |in_int|` as u32 (`-2^31 -> 0x80000000` exact).
2. LZC32 -> `lz`; `norm = mag << lz` puts MSB at bit31.
3. `sig8 = norm[31:24]`, `G = norm[23]`, `S = |norm[22:0]|`;
   `up = G & (sig8[0] | S)`; round-carry bumps `E`: `E = (31 - lz) + rc`, `sig8 = 0x80`.
   Result `r8 = ±sig8 * 2^(E-7)`, biased exponent `E + 127`. Zero flag passes through.

**Cut (1 mid register, payload {sign, zero, sig8, E, scale}):** remaining
`L_CONV - 1` registers chain on the output bf16. RMW instantiates `L_CONV=1` ->
front half / back half land in adjacent stages (S1/S2), matching Phase 2c's stage map.

**Back half — scale, denormalize, encode (second RNE where needed):**
1. `e_tot = E + signed(scale)` — biased result exponent, 10-bit signed
   (E in [0,32], scale in [-256,255] -> e_tot in [-256,287]; no wrap anywhere —
   this natively subsumes v2's flush-vs-wrap fix).
2. `zero -> 16'h0000`. `e_tot >= 255 -> {sign, FF, 0}` (inf saturation; golden fp32
   overflow -> inf -> bf16 inf). `e_tot in [1,254] -> {sign, e_tot[7:0], sig8[6:0]}`
   — **exact**, the scale shift never re-rounds a normal result.
3. `e_tot <= 0` -> subnormal: shift `s = 1 - e_tot` >= 1 through the same 11-bit
   `{sig8,G,R,S}` funnel as the adder (`min(s,11)`, sticky-OR), RNE round to the 7-bit
   subnormal frac; `f8 = 0x80` promotes to min normal, `f8 = 0` is the signed-zero
   flush (deep underflow rounds to ±0 automatically — s >= 9 makes G=0).
   This single rounder reproduces golden's ldexp -> bf16 chain including the
   tie at 2^-134 -> +0 (ties-to-even) and 1.5*2^-134 -> min subnormal.

## 4. Oracle strengthening (add, never weaken)

- `gen_bf16_add` directed edges (+~16): finite overflow `(max,max) -> inf`,
  `(-max,-max)`, near-overflow rounding pairs, min-normal minus min-subnormal
  (cross-boundary borrow -> `0x007F`), min-normal + min-subnormal (exact `0x0081`),
  `1.0 ± 2^-24` (d=24 sticky-collapse; the minus case exercises sub + shift +
  round-carry back to 1.0), `x + (-x) -> +0` (incl. subnormal), `(-0)+(-0) -> -0`,
  RNE ties `1.0 + 2^-8 -> 1.0` (even) and `(1+2^-7) + 2^-8 -> round up`.
- `gen_int_to_bf16` directed ints (+48): full-int32-range magnitudes
  `{2^31-1, -2^31, 2^31-65, -(2^31-65), 0x40000001, -(0x40000001)}` x 8 scales
  (front-half LZC/round at maximum magnitude; realizable-band vectors unchanged).

Counts change (docs updated accordingly): bf16_add 200005 -> 200021,
int_to_bf16 32312 -> 32360. Sentinels are count-agnostic (`ALL .* TESTS PASSED`).

## 5. File changes

| File | Change |
|---|---|
| `gemm_sram.srcs/sources_1/new/bf16_adder.v` | Full native rewrite (§2). |
| `gemm_sram.srcs/sources_1/new/int_to_bf16.v` | Full native rewrite (§3). |
| `sim/bf16_vectors.py` | §4 directed vectors. |
| `tb/bf16_adder_tb.v`, `tb/int_to_bf16_tb.v`, `tb/rmw_tb.v` | Header text only (implementation description); logic untouched. |
| `sim/run_bf16_adder.sh` | Compile list -> `bf16_adder.v` + TB only. |
| `sim/run_int_to_bf16.sh` | Compile list -> `int_to_bf16.v` + TB only. |
| `sim/run_rmw.sh` | Compile list -> `int_to_bf16.v bf16_adder.v RMW.v` + TB. |
| `sim/run_top_elab.sh`, `sim/run_integration_one.sh`, `sim/run_integration_smoke.sh`, `sim/run_mixed_one.sh` | Drop `HardFloatBundle*.v`, `fp32_adder.v`, `fp32_to_bf16_rne.v` (datapath is HardFloat-free; elab failure would expose any remaining reference). |
| `work/synth_rmw/synth.tcl` | bf16 mode reads only `int_to_bf16.v bf16_adder.v RMW.v`. |
| `RMW.v` | Header stage map text only (S1..S5 now native block names). |
| `fp32_to_bf16_rne.v` | Header status ACTIVE -> PRESERVED (no longer instantiated by the datapath; own 70012 gate stays). |
| `rtl/README.md`, `gemm_sram.srcs/sources_1/new/README.md`, `rtl/refresh.sh` copies | Status/diagram/table updates + new synth numbers. |
| fp32 unit scripts (`run_fp32_adder.sh` etc.), HardFloat bundles, `fp32_adder.v`, `int_to_fp32.v` | **Untouched** (preserved line + recovery tag). |

## 6. Gates (all must be green before merge)

1. `bash sim/run_bf16_adder.sh` -> `ALL 200021 TESTS PASSED` (dual-DUT: default + RMW config)
2. `bash sim/run_int_to_bf16.sh` -> `ALL 32360 TESTS PASSED` (dual-DUT: L_CONV 2 + 1)
3. `cd MXP_Tools && python -m mxp_tools rmw-gen --out work/rmw --n 64 --seed 0` then
   `bash sim/run_rmw.sh` -> `ALL 113 TESTS PASSED` (sole skew/latency gate)
4. `bash sim/run_top_elab.sh` -> elab clean (proves HardFloat-free hierarchy)
5. `bash sim/run_integration_sweep.sh` -> `ALL 9 MODES PASSED` (bit-exact vs bf16 golden)
6. `python sim/runner.py mixed-sweep` -> `ALL 3 MIXED MODES PASSED`
7. Preserved units no-regression: `run_fp32_to_bf16_rne.sh` (70012), `run_fp32_adder.sh`, `run_int_to_fp32.sh`
8. Synth proxy: `vivado -mode batch -nojournal -nolog -source work/synth_rmw/synth.tcl -tclargs bf16`
   (flags BEFORE -tclargs) -> record LUT/FF/WNS; expect WNS >> -8.79, target >= 0 @ 4ns pre-place.

## 7. Risks / arbitration

- [SEMANTICS] New rounding arithmetic. Known hard spots, each with a directed-vector
  witness: sticky-through-subtraction (GRS OR-before-sub), cancellation-vs-sticky
  left-shift invariant, subnormal clamp in normalize, round-carry renormalization,
  zero-sign rules, denormalization double-round (golden's own semantic), inf
  saturation, tie-to-even at 2^-134.
- Arbitration: the ml_dtypes oracles + rmw_tb + integration golden are the truth.
  Any red gate = RTL bug. Vectors may only be added.
- Review per project rule: comprehensive + adversarial reviewer (>=1) after the chunk.

## 8. Task checklist

- [ ] T1: plan committed
- [ ] T2: native `bf16_adder.v` + adder directed vectors + `run_bf16_adder.sh` trim + TB header; gate 1 green
- [ ] T3: native `int_to_bf16.v` + converter directed vectors + `run_int_to_bf16.sh` trim + TB header; gate 2 green
- [ ] T4: `run_rmw.sh`/integration/synth script trims + `RMW.v`/`fp32_to_bf16_rne.v` headers + `rmw_tb.v` header; gates 3-7 green
- [ ] T5: synth measurement (gate 8) + README/rtl-copy refresh + docs
- [ ] T6: reviews (comprehensive + adversarial) -> fixes -> converge
- [ ] T7: merge to main (ff), branch delete, no push; CLAUDE.md kickoff + memory update
