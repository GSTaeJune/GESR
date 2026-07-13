# RMW Pipeline Rebalance (Phase 2c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redistribute the RMW unit's 5 pipeline registers so that each of the five major FP combinational blocks (INToRecFN / FNFromRecFN+narrow / RecFNFromFN / AddRecFN / FNFromRecFN+narrow) gets its own stage, and out_RMW becomes a registered output — total latency stays exactly 5 cycles, function stays bit-exact (pure register relocation on a feedforward path).

**Architecture:** Today all 3 fp32_adder regs sit on the recoded OPERANDS before AddRecFN, and int_to_bf16's 2 regs sit back-to-back after its exp-add — so 3 of 5 stages are pure wire-shifts while two reg-to-reg spans each contain a full FP normalize+round chain (Span B: FNFromRecFN+narrow+widen+RecFNFromFN; Span C: AddRecFN+FNFromRecFN+narrow ending UNREGISTERED at the SRAM D input). The fix: parameterize stage positions in bf16_adder (L_IN/L_SUM/L_OUT) and fp32_adder (L_SUM), then have RMW instantiate the balanced arrangement. All new parameters default to current behavior, so every existing unit TB stays green unchanged; the new arrangement is additionally covered by dual-DUT instances in two unit TBs (full 32312 + 200005 vector sets through the new register placement) plus rmw_tb (113) and the integration sweeps.

**Tech Stack:** Verilog-2001, XSim (bash sim/run_*.sh), Vivado OOC synth proxy (work/synth_rmw/synth.tcl), MXP_Tools golden vectors.

**Provenance:** Curated finding "T2" from the 2026-07-12 nine-agent RTL analysis + fable adversarial curation (this session). Scope deliberately excludes: SRAM/BRAM items (SRAM RTL is sim-only; real SRAM = CACTI/foundry macro), all `imports/` (MXP core) files, all `third_party/` generated HardFloat files, and the AddRecFN-internal split (T3 — only if this rebalance proves insufficient later).

**Backend context:** The user will PNR this RTL (ASIC). Vivado numbers are a relative proxy only. The rebalance is target-independent (register placement is an RTL structural property); the registered out_RMW also removes the combinational tail into the SRAM macro's D-input setup path, which matters for PNR.

---

## Pipeline: before / after (RMW config L_CONV=2, L_ADD=3, total 5)

```
BEFORE (regs marked #, spans labeled):
 in_GEMM,scale --[INToRecFN + exp-add/flush]--#1--(wire)--#2--[FNFromRecFN + narrow]--+
 in_SRAM ------------------------#s1--(wire)--#s2-----------------------------------+
   +--[widen + RecFNFromFN x2]--#3--(wire)--#4--(wire)--#5--[AddRecFN + FNFromRecFN + narrow]--> out_RMW (COMB out)
 Span B = #2 -> #3 : FNFromRecFN + narrow + widen + RecFNFromFN   (monster)
 Span C = #5 -> SRAM D : AddRecFN + FNFromRecFN + narrow + D-mux  (monster, unregistered out)

AFTER:
 S1: [INToRecFN + exp-add/flush]           -> #1  (int_to_bf16, L_CONV-1 = 1 reg)
 S2: [FNFromRecFN + narrow] (i2b tail)     -> #2  (bf16_adder L_IN reg, a & b both, 16b)
 S3: [widen(wire) + RecFNFromFN x2]        -> #3  (fp32_adder operand regs, L_ADD-2 = 1)
 S4: [AddRecFN]                            -> #4  (fp32_adder L_SUM reg, recoded sum 33b)
 S5: [FNFromRecFN + narrow] (adder tail)   -> #5  (bf16_adder L_OUT reg, 16b)
 out_RMW = #5 output (REGISTERED) -> D-mux -> SRAM D
 in_SRAM alignment: sram_dly depth L_CONV-1 = 1 (meets fp_a at the bf16_adder input;
   the shared L_IN reg then delays a and b identically, preserving pairing).
```

Latency bookkeeping: (L_CONV-1) + L_IN(1) + (L_ADD-2) + L_SUM(1) + L_OUT(1) = L_CONV + L_ADD = 5. External RMW parameter semantics (total = L_CONV + L_ADD) unchanged -> `rmw_tb` (passes 2/3) and `gemm_sram_top` (uses defaults) need no edits.

**Why this is bit-exact by construction:** every change is a register inserted/removed on an existing feedforward net (no logic modified, no feedback, rst unused on these regs, per-cycle pipelining preserved because every stage is a simple full-width reg). The arithmetic (INToRecFN -> exp-add -> flush -> FNFromRecFN -> narrow -> widen -> RecFNFromFN -> AddRecFN -> FNFromRecFN -> narrow) is byte-for-byte the same netlist function, just sampled at different stage boundaries. Gates arbitrate anyway (32312 / 200005 / 113 / 9-mode / mixed).

**Parameter surface (all defaults = current behavior):**

| Module | Params (new in bold) | Default | RMW config |
|---|---|---|---|
| int_to_bf16 | L_CONV | 2 | 1 (via L_CONV-1) |
| fp32_adder | L_ADD, **L_SUM** | 3, 0 | 1, 1 |
| bf16_adder | **L_IN**, L_ADD, **L_SUM**, **L_OUT** | 0, 3, 0, 0 | 1, 1, 1, 1 |
| RMW | L_CONV, L_ADD (semantics unchanged: total = sum) | 2, 3 | (constraint now L_CONV>=2, L_ADD>=3) |

Note on bf16_adder.L_ADD semantic shift: it previously meant "total adder latency"; now it means "fp32_adder operand-side stages". Numerically identical at defaults (L_IN=L_SUM=L_OUT=0 -> total = L_ADD), so `bf16_adder_tb`'s existing instance and comments stay valid; headers must document the new meaning.

---

### Task 0: Branch

**Files:** none (git only)

- [ ] **Step 0.1:** From repo root (must be on `main`, clean tree):

```bash
git checkout -b feat/rmw-pipeline-rebalance
```

- [ ] **Step 0.2 (baseline for honest delta):** NO re-run needed. The existing artifacts under `work/synth_rmw/bf16/` (`timing.rpt` WNS=-2.547, `util.rpt` 768 Slice LUTs of which 34 are LUT-as-shift-register, 99 FF) plus the logged `SYNTH_SUMMARY mode=bf16 WNS=-2.547 LUT_CELLS=879 FF_CELLS=99` were produced by this exact script on RTL that is functionally identical to HEAD (only comment-only commits since). Metric note (reviewer-verified): 768 = util.rpt "Slice LUTs", 879 = `get_cells PRIMITIVE_GROUP==LUT` cell count — two countings of the same run, both valid; compare like against like in Task 6. Script usage note: `synth.tcl` takes ONE positional mode via `-tclargs <mode>` and errors (`ERROR=unknown-mode`, exit 1) without it — it does NOT iterate modes.

- [ ] **Step 0.3:** Commit nothing yet (synth artifacts live under gitignored `work/`).

---

### Task 1: fp32_adder.v — add L_SUM result-side stage

**Files:**
- Modify: `gemm_sram.srcs/sources_1/new/fp32_adder.v`

- [ ] **Step 1.1:** Change the module parameter list and header. Parameter block becomes:

```verilog
module fp32_adder #(
    parameter L_ADD = 3,           // 피연산자단(recoded) 파이프라인 단수 (>=1)
    parameter L_SUM = 0            // 합(recoded)단 파이프라인 단수 (>=0, AddRecFN 뒤)
)(
```

Header comment "파이프라인 구조" section: state that L_ADD regs sit on the recoded operands (as today) and L_SUM regs sit on the recoded sum between AddRecFN and FNFromRecFN; total latency = L_ADD + L_SUM; defaults (3,0) reproduce the previous fixed behavior.

- [ ] **Step 1.2:** Insert the L_SUM stage between AddRecFN and FNFromRecFN. After the existing `AddRecFN u_add (...)` block (which drives `recFN_sum`), replace the final `FNFromRecFN_wrapper u_out (.in(recFN_sum), ...)` connection with:

```verilog
    // 3.5) L_SUM단 결과 레지스터 (recoded 합 위에; L_SUM=0 이면 조합 통과)
    wire [REC_W-1:0] recFN_sum_dly;
    generate
        if (L_SUM > 0) begin : g_sum_reg
            reg [REC_W-1:0] sum_dly [0:L_SUM-1];
            integer si;
            always @(posedge clk) begin
                sum_dly[0] <= recFN_sum;
                for (si = 1; si < L_SUM; si = si + 1)
                    sum_dly[si] <= sum_dly[si-1];
            end
            assign recFN_sum_dly = sum_dly[L_SUM-1];
        end else begin : g_sum_comb
            assign recFN_sum_dly = recFN_sum;
        end
    endgenerate

    // 4) recoded -> IEEE-754 FP32 환원 (조합 회로, 출력단에서)
    FNFromRecFN_wrapper u_out (
        .in  (recFN_sum_dly),
        .out (sum)
    );
```

- [ ] **Step 1.3:** Run the fp32 unit gate (defaults 3,0 -> behavior identical):

```bash
bash sim/run_fp32_adder.sh
```

Expected: the script's pass sentinel (fp32_adder_tb PASS line) — unchanged from HEAD.

- [ ] **Step 1.4:** Commit:

```bash
git add gemm_sram.srcs/sources_1/new/fp32_adder.v
git commit -m "feat(rmw-rebalance): fp32_adder L_SUM result-side stage (default 0 = no change)"
```

---

### Task 2: bf16_adder.v — L_IN / L_SUM / L_OUT stages + dual-DUT TB

**Files:**
- Modify: `gemm_sram.srcs/sources_1/new/bf16_adder.v`
- Modify: `tb/bf16_adder_tb.v`

- [ ] **Step 2.1:** Rewrite bf16_adder with the new parameter surface. Full new body (header comment: keep the existing 검증 목적/NaN notes, add the stage-parameter table and the L_ADD semantic note):

```verilog
module bf16_adder #(
    parameter L_IN  = 0,   // bf16 입력단 레지스터 (a/b 공통 지연, >=0)
    parameter L_ADD = 3,   // fp32_adder 피연산자단 (recoded, >=1) — 종전 "전체 지연" 의미에서 변경
    parameter L_SUM = 0,   // fp32_adder 합(recoded)단 (>=0)
    parameter L_OUT = 0    // bf16 출력단 (narrow 후, >=0)
)(                          // 전체 지연 = L_IN + L_ADD + L_SUM + L_OUT
    input  wire        clk,
    input  wire        rst,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] sum
);
    // 0) L_IN단 bf16 입력 레지스터. a/b 를 같은 체인 길이로 지연하므로
    //    두 피연산자의 상호 정렬(pairing)은 불변.
    wire [15:0] a_d;
    wire [15:0] b_d;
    generate
        if (L_IN > 0) begin : g_in_reg
            reg [15:0] a_q [0:L_IN-1];
            reg [15:0] b_q [0:L_IN-1];
            integer ii;
            always @(posedge clk) begin
                a_q[0] <= a;
                b_q[0] <= b;
                for (ii = 1; ii < L_IN; ii = ii + 1) begin
                    a_q[ii] <= a_q[ii-1];
                    b_q[ii] <= b_q[ii-1];
                end
            end
            assign a_d = a_q[L_IN-1];
            assign b_d = b_q[L_IN-1];
        end else begin : g_in_comb
            assign a_d = a;
            assign b_d = b;
        end
    endgenerate

    // 1) exact widen -> 2) fp32 도메인 가산 (L_ADD 피연산자단 + L_SUM 합단)
    wire [31:0] a_fp32 = {a_d, 16'h0000};
    wire [31:0] b_fp32 = {b_d, 16'h0000};
    wire [31:0] sum_fp32;
    fp32_adder #(.L_ADD(L_ADD), .L_SUM(L_SUM)) u_add (
        .clk(clk), .rst(rst), .a(a_fp32), .b(b_fp32), .sum(sum_fp32)
    );

    // 3) 단일 RNE narrow (조합)
    wire [15:0] sum_bf16;
    fp32_to_bf16_rne u_narrow (.in(sum_fp32), .out(sum_bf16));

    // 4) L_OUT단 bf16 출력 레지스터 (RMW 구성에서 out_RMW 를 레지스터 출력으로
    //    만들어 SRAM D 입력까지의 조합 꼬리를 제거한다)
    generate
        if (L_OUT > 0) begin : g_out_reg
            reg [15:0] sum_q [0:L_OUT-1];
            integer oi;
            always @(posedge clk) begin
                sum_q[0] <= sum_bf16;
                for (oi = 1; oi < L_OUT; oi = oi + 1)
                    sum_q[oi] <= sum_q[oi-1];
            end
            assign sum = sum_q[L_OUT-1];
        end else begin : g_out_comb
            assign sum = sum_bf16;
        end
    endgenerate
endmodule
```

- [ ] **Step 2.2:** Extend `tb/bf16_adder_tb.v` with a second DUT in the exact RMW configuration, so all 200005 vectors also traverse the new register arrangement. Changes (keep existing header conventions; add a 검증 내용 bullet about the dual-DUT):

After `localparam L_ADD = 3;` add:

```verilog
    // RMW 실배치 구성 (Phase 2c 재배치): dut_rmw 의 L_IN+L_ADD+L_SUM+L_OUT = 1+1+1+1.
    // 주의: 아래 dut_rmw 인스턴스 파라미터를 바꾸면 이 값도 함께 갱신할 것.
    localparam L_RMW  = 4;
    localparam L_WAIT = (L_RMW > L_ADD) ? L_RMW : L_ADD;   // 두 DUT 모두 통과할 대기
```

After the existing `dut` instance add:

```verilog
    wire [15:0] sum_rmw;
    bf16_adder #(.L_IN(1), .L_ADD(1), .L_SUM(1), .L_OUT(1)) dut_rmw (
        .clk (clk),
        .rst (rst),
        .a   (a),
        .b   (b),
        .sum (sum_rmw)
    );
```

In the vector loop, change the wait bound from `L_ADD + 1` to `L_WAIT + 1` (inputs are held for the whole wait, so the shorter-latency DUT's output is settled and stable at sample time), and duplicate the compare for `sum_rmw`:

```verilog
            for (k = 0; k < L_WAIT + 1; k = k + 1)
                @(posedge clk);
            #1;  // let combinational narrow settle after the last posedge
            // 둘 다 NaN 이면 부호/payload 무시하고 통과, 그 외는 bit-exact.
            if (!(is_bf16_nan(sum) && is_bf16_nan(exp_w)) && (sum !== exp_w)) begin
                if (nfail < 10)
                    $display("MISMATCH(dut)     a=%04x b=%04x got=%04x exp=%04x",
                             a_w, b_w, sum, exp_w);
                nfail = nfail + 1;
            end
            if (!(is_bf16_nan(sum_rmw) && is_bf16_nan(exp_w)) && (sum_rmw !== exp_w)) begin
                if (nfail < 10)
                    $display("MISMATCH(dut_rmw) a=%04x b=%04x got=%04x exp=%04x",
                             a_w, b_w, sum_rmw, exp_w);
                nfail = nfail + 1;
            end
```

`ntot` stays incremented once per vector, so the pass sentinel remains `ALL 200005 TESTS PASSED` (run script grep unchanged). **Intentional design (do not "fix"):** `nfail` is SHARED across both DUT compares (OR semantics) — any mismatch on either DUT makes `nfail != 0` and suppresses the single PASS sentinel. `ntot` counts vectors, not comparisons; splitting into per-DUT counters/sentinels would break the run-script grep contract.

- [ ] **Step 2.3:** Run the gate:

```bash
bash sim/run_bf16_adder.sh
```

Expected: `bf16_adder_tb: ALL 200005 TESTS PASSED` (now covering BOTH configurations).

- [ ] **Step 2.4:** Commit:

```bash
git add gemm_sram.srcs/sources_1/new/bf16_adder.v tb/bf16_adder_tb.v
git commit -m "feat(rmw-rebalance): bf16_adder L_IN/L_SUM/L_OUT stages + dual-DUT TB (200005 x2 configs)"
```

---

### Task 3: int_to_bf16_tb — dual-DUT at L_CONV=1

**Files:**
- Modify: `tb/int_to_bf16_tb.v` (RTL `int_to_bf16.v` itself is UNCHANGED — the L_CONV param already exists)

- [ ] **Step 3.1:** Add a second DUT with the RMW-config depth. After the existing `dut` instance:

```verilog
    // RMW 재배치 구성 (Phase 2c): int_to_bf16 은 L_CONV-1 = 1 단으로 인스턴스된다.
    wire [15:0] out_bf16_l1;
    int_to_bf16 #(.L_CONV(1)) dut_l1 (
        .clk      (clk),
        .rst      (rst),
        .in_int   (in_int),
        .scale    (scale),
        .out_bf16 (out_bf16_l1)
    );
```

The existing wait (`L_CONV + 1 = 3` posedges with inputs held) already covers the depth-1 instance (settled and stable). Duplicate the compare:

```verilog
            if (out_bf16 !== exp_w) begin
                if (nfail < 10)
                    $display("MISMATCH(L2) in_int=%08x scale=%03x got=%04x exp=%04x",
                             in_int_w, scale_w, out_bf16, exp_w);
                nfail = nfail + 1;
            end
            if (out_bf16_l1 !== exp_w) begin
                if (nfail < 10)
                    $display("MISMATCH(L1) in_int=%08x scale=%03x got=%04x exp=%04x",
                             in_int_w, scale_w, out_bf16_l1, exp_w);
                nfail = nfail + 1;
            end
```

Header: add one 검증 내용 bullet noting the dual-DUT (L_CONV=2 기본 + L_CONV=1 RMW 구성). `ntot` unchanged -> sentinel stays `ALL 32312 TESTS PASSED`. Same intentional shared-`nfail` design as Task 2 (OR across both DUTs, single sentinel).

- [ ] **Step 3.2:** Run the gate (vectors must exist; regenerate if `work/bf16_vec/` is missing):

```bash
bash sim/run_int_to_bf16.sh
```

Expected: `int_to_bf16_tb: ALL 32312 TESTS PASSED`.

- [ ] **Step 3.3:** Commit:

```bash
git add tb/int_to_bf16_tb.v
git commit -m "test(rmw-rebalance): int_to_bf16_tb dual-DUT (L_CONV=2 default + L_CONV=1 RMW config)"
```

---

### Task 4: RMW.v — balanced instantiation

**Files:**
- Modify: `gemm_sram.srcs/sources_1/new/RMW.v`

- [ ] **Step 4.1:** Update the parameter constraint check (message ASCII-only per project rule):

```verilog
    // 파라미터 안전 체크: 재배치 파이프라인은 변환측 >=2, 가산측 >=3 을 요구한다
    // (내부 분배: int_to_bf16 L_CONV-1 / L_IN 1 / 피연산자 L_ADD-2 / L_SUM 1 / L_OUT 1).
    initial begin
        if (L_CONV < 2 || L_ADD < 3) begin
            $display("FATAL RMW: L_CONV=%0d L_ADD=%0d (need L_CONV>=2 && L_ADD>=3)",
                     L_CONV, L_ADD);
            $finish;
        end
    end
```

- [ ] **Step 4.2:** Rewire the three internal pieces:

```verilog
    // 1) INT32 -> bf16 디퀀타이즈. 재배치: 변환기 내부 레지스터는 L_CONV-1 단,
    //    남은 1단은 bf16_adder 의 L_IN(입력단) 레지스터가 담당한다.
    wire [15:0] fp_a;
    int_to_bf16 #(.L_CONV(L_CONV-1)) u_conv (
        .clk      (clk),
        .rst      (rst),
        .in_int   (in_GEMM),
        .scale    (scale),
        .out_bf16 (fp_a)
    );

    // 2) in_SRAM 을 L_CONV-1 사이클 늦춰서 fp_a 와 가산기 "입력에서" 정렬시킨다.
    //    (L_IN 레지스터는 a/b 를 함께 지연하므로 여기서는 변환기 내부 단수만 맞춘다.)
    reg [15:0] sram_dly [0:L_CONV-2];
    integer di;
    always @(posedge clk) begin
        sram_dly[0] <= in_SRAM;
        for (di = 1; di < L_CONV-1; di = di + 1)
            sram_dly[di] <= sram_dly[di-1];
    end

    // 3) bf16 덧셈. 재배치 구성: 입력 1 + 피연산자 L_ADD-2 + 합 1 + 출력 1 단.
    //    L_OUT=1 이 out_RMW 를 레지스터 출력으로 만들어 SRAM D 까지의 조합 꼬리를
    //    제거한다 (PNR 시 매크로 D-setup 경로 단축).
    bf16_adder #(.L_IN(1), .L_ADD(L_ADD-2), .L_SUM(1), .L_OUT(1)) u_add (
        .clk (clk),
        .rst (rst),
        .a   (fp_a),
        .b   (sram_dly[L_CONV-2]),
        .sum (out_RMW)
    );
```

- [ ] **Step 4.3:** Update the header comment: replace the 데이터 패스 diagram + "전체 파이프라인 지연" note with the AFTER stage map from this plan (S1..S5 + registered out_RMW + sram_dly 정렬 논리). Keep total = L_CONV + L_ADD wording so the TB-contract sentence stays true.

- [ ] **Step 4.4:** Regenerate RMW vectors if needed and run the gate:

```bash
cd MXP_Tools && python -m mxp_tools rmw-gen --out work/rmw --n 64 --seed 0 && cd ..
bash sim/run_rmw.sh
```

Expected: `rmw_tb: ALL 113 TESTS PASSED` (rmw_tb passes .L_CONV(2)/.L_ADD(3) — same totals, no TB edit).

**Coverage statement (load-bearing, record in header/docs):** rmw_tb's back-to-back streaming (1 vector/cycle, per-index pure-function oracle with NO pipeline model) is the SOLE gate that detects a/b alignment skew inside RMW — a mis-depthed sram_dly pairs in_SRAM[i] with the wrong dq[i+-1] and fails almost every vector. The integration sweep CANNOT catch skew: its per-col drain FSM holds both operands static for the whole 10-cycle window (in_GEMM held reg; Q frozen by CEB=1). Do NOT weaken rmw_tb's streaming cadence in any future "simplification".

- [ ] **Step 4.5:** Commit:

```bash
git add gemm_sram.srcs/sources_1/new/RMW.v
git commit -m "feat(rmw-rebalance): balanced 5-stage instantiation (1 reg per FP block, registered out_RMW)"
```

---

### Task 5: Integration gates

**Files:** none (verification only)

- [ ] **Step 5.1:** Elab smoke:

```bash
bash sim/run_top_elab.sh
```

Expected: elab clean (gemm_sram_top instantiates `RMW u_rmw` with defaults 2/3 — unchanged).

- [ ] **Step 5.2:** Full 9-mode integration sweep (bit-exact vs bf16 golden):

```bash
bash sim/run_integration_sweep.sh
```

Expected last line: `ALL 9 MODES PASSED` (~10.5 min).

- [ ] **Step 5.3:** Mixed-precision sweep:

```bash
python sim/runner.py mixed-sweep
```

Expected: `ALL 3 MIXED MODES PASSED`.

- [ ] **Step 5.4:** Remaining unit gates untouched by the change (regression sanity):

```bash
bash sim/run_fp32_to_bf16_rne.sh   # 70012
bash sim/run_int_to_fp32.sh        # fp32 unit TB (preserved module, unchanged)
```

Expected: both pass sentinels unchanged.

- [ ] **Step 5.5:** No commit (nothing changed).

---

### Task 6: Synth re-measure (proxy evidence)

**Files:**
- Possibly modify: none (script reused as-is)

- [ ] **Step 6.1:** Run the OOC script on the branch head (bf16 mode; the script synthesizes RMW top with DEFAULT params 2/3, which now elaborate the balanced arrangement — reviewer-verified, so the delta is real, not zero-by-construction):

```bash
mkdir -p work/synth_rmw/logs
vivado -mode batch -source work/synth_rmw/synth.tcl -tclargs bf16 -nojournal -nolog > work/synth_rmw/logs/rebalance_post.log 2>&1 || true
grep "SYNTH_SUMMARY" work/synth_rmw/logs/rebalance_post.log
```

(If bash cannot find vivado, use the shell the previous session used; artifacts land in `work/synth_rmw/bf16/`.)

- [ ] **Step 6.2:** Record before/after vs the Step 0.2 baseline (`WNS=-2.547 LUT_CELLS=879 FF_CELLS=99`; util.rpt 768 Slice LUTs / 99 FF): WNS, LUT_CELLS, FF_CELLS — compare like metric against like. Expected direction: WNS improves substantially toward 0 at the 4ns constraint (AddRecFN alone becomes the longest stage); LUT ~flat; FF direction NOT pre-committed — register-width accounting suggests FF likely DOWN (two 2x33b operand stages removed vs narrow 16b stages + one 33b sum stage added), but let SYNTH_SUMMARY decide and report the measured number. If WNS does NOT improve, that is a real finding: report it, do not spin — the RTL change is still correct (gates green) and the fallback (T3 AddRecFN split) becomes the next decision, made by the human.

- [ ] **Step 6.3:** No commit for logs (gitignored). Numbers go into Task 7's docs.

---

### Task 7: Docs + reading-copy refresh

**Files:**
- Modify: `gemm_sram.srcs/sources_1/new/README.md` (pipeline description + stage map)
- Modify: `rtl/README.md` (synth table: add rebalance row with Task 6 numbers)
- Run: `bash rtl/refresh.sh` (re-copy changed RTL into the reading folder)
- Modify: `CLAUDE.md` (short Phase 2c note in the newest kickoff section: what changed, total latency unchanged, gates re-verified, synth delta)

- [ ] **Step 7.1:** Update the docs with: the S1..S5 stage map, "total 5cy unchanged / TB contract intact", the rmw_tb-is-the-sole-skew-gate coverage statement (Task 4), dual-DUT TB coverage note, and the measured synth delta. Two consistency obligations (adversarial-review findings):
  - `CLAUDE.md` "RMW contract" prose and the Settled table row "RMW (L_CONV, L_ADD) = (2, 3), total 5 cyc" must gain one sentence: L_CONV/L_ADD remain the EXTERNAL knobs (sum = total = 5, TB contract unchanged) but the INTERNAL per-block register distribution changed in Phase 2c (S1..S5 — int_to_bf16 now holds 1 reg, bf16_adder holds 4 across L_IN/operand/L_SUM/L_OUT). Otherwise a future reader debugs assuming 2 regs live inside int_to_bf16.
  - `rtl/README.md` synth table: if Task 6 synth was skipped, annotate the existing 768 LUT / ~153MHz row as "pre-rebalance" rather than silently retaining it as current.

- [ ] **Step 7.2:** `bash rtl/refresh.sh` then verify `git status` shows only intended rtl/ copies changed.

- [ ] **Step 7.3:** Commit:

```bash
git add gemm_sram.srcs/sources_1/new/README.md rtl/ CLAUDE.md
git commit -m "docs(rmw-rebalance): stage map, synth delta, reading-copy refresh"
```

---

### Task 8: Reviews to convergence, then merge

- [ ] **Step 8.1:** Dispatch code review per project rule: comprehensive reviewer + adversarial reviewer (>=1, standing user rule) on the full branch diff (`main..HEAD`). Adversarial focus: (a) any stage where a register move could change sampling of a signal that is NOT purely feedforward; (b) TB dual-DUT wait-bound correctness (held inputs, settled outputs); (c) generate-block legality under XSim Verilog-2001; (d) false-green risks in the pass sentinels.
- [ ] **Step 8.2:** Fix findings, re-run affected gates, iterate until no blocking issues.
- [ ] **Step 8.3:** Use superpowers:finishing-a-development-branch — expected outcome per project convention: local ff-merge to `main`, delete branch, NO push (push only on explicit request).

---

## Outcome (2026-07-13, recorded at Task 6/7)

**All gates green, zero deviations from plan code blocks:** fp32_adder PASS / bf16_adder 200005 (both configs via dual-DUT) / int_to_bf16 32312 (L_CONV 2 and 1) / rmw 113 / top elab / fp32_to_bf16_rne 70012 / int_to_fp32 PASS / integration ALL 9 MODES PASSED (bit-exact) / mixed ALL 3 PASSED. Commits 655682b, dc65f25, e633150, 8e877d0.

**Synth (OOC K7-160T, 4ns) — the honest-measurement finding:**
- Rebalanced: `SYNTH_SUMMARY mode=bf16 WNS=-8.793 LUT_CELLS=866 FF_CELLS=148`; util 744 Slice LUTs (SRL 0) / 148 FF. Critical path = S4 stage alone (`recFN_a_dly[0] -> AddRecFN -> g_sum_reg.sum_dly[0]`), est. 12.36ns = logic 4.09 (29 levels) + route 8.27 (pre-place estimate).
- Baseline `-2.547 (~153MHz)` was a HALF-MEASUREMENT: out_RMW was combinational-to-port, so the whole AddRecFN(+FNFromRecFN+narrow) path sat in the `no_output_delay` (untimed) group — the script sets only `create_clock`, no input/output delays (57 no_input_delay / 16 no_output_delay ports confirmed in timing.rpt). Baseline WNS timed only Span B. The two WNS numbers must NOT be compared directly.
- Step 6.2's "expect WNS improves substantially" premise was therefore flawed — the plan's hedge branch applies: RTL is correct (all gates green), the rebalance made timing measurement honest and removed the SRAM-D combinational tail (registered out_RMW), and the true bottleneck is now proven to be the single-stage AddRecFN. **250MHz requires the T3 AddRecFN internal split — next decision belongs to the human.**
- Ops gotcha recorded: `-nojournal -nolog` must precede `-tclargs`, else they are consumed as the script's part argument (`synth_design -part -nojournal` error; first post-change run failed this way and left stale baseline reports in place — detected via identical-to-baseline numbers + commented-out SYNTH_SUMMARY grep hit).
