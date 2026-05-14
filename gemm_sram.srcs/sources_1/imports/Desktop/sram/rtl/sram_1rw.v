//=============================================================================
// SRAM 1RW Bank - Behavioral RTL (Verilog-2001)
//
// Single-port (1RW) synchronous SRAM. "1RW"는 한 cycle에 read 또는 write
// 둘 중 하나만 할 수 있는 single-port 구조를 의미. 실제 chip에서 가장 흔히
// 쓰이는 SRAM topology (가장 작은 6T bitcell 사용).
//
// 설계 의도: 이 behavioral RTL은 memory compiler가 뽑아내는 foundry SRAM
// macro와 외부 interface 및 timing이 동일하다. 즉 PNR 단계에서 이 module을
// 실제 macro로 swap해도, 이걸 instantiate하는 상위 code는 변경할 필요가
// 없다.
//
// Spec:  docs/superpowers/specs/2026-05-13-sram-1rw-bank-design.md
// Style: Pure Verilog-2001 (no SystemVerilog 기능 사용 금지).
//=============================================================================

// `timescale은 Vivado XSim의 규칙("design 안에 timescale이 있는 module이
// 하나라도 있으면 모든 module에 있어야 함")을 만족시키기 위해서만 존재한다.
// Synthesis tool은 이 directive를 완전히 무시 — 순수하게 simulation 시간
// 단위에만 영향:
//   1ns = #delay 표현식의 단위 (예: #5 = 5 ns 지연)
//   1ps = simulator 내부 time precision
`timescale 1ns/1ps

//-----------------------------------------------------------------------------
// Module 선언은 "non-ANSI" port style (Verilog-95 시절 방식) 사용:
//   - Port list (CLK, CEB, ...)에는 signal 이름만 나열.
//   - 방향(input/output)과 width는 body에서 따로 선언.
//
// 왜 non-ANSI를 쓰나? Port `A`의 width 선언에 `ADDR_WIDTH`를 써야 하는데,
// `ADDR_WIDTH`는 `clog2(DEPTH)`로 derive해야 하고, `clog2` 함수는 module
// body 안에서 정의되기 때문. ANSI 스타일(현대식)은 header의 parameter가
// body의 함수를 참조 못 함 — 닭달걀 문제. Non-ANSI는 함수 먼저 정의 →
// localparam으로 ADDR_WIDTH 계산 → 그 width로 port 선언 순서가 가능.
//-----------------------------------------------------------------------------
module sram_1rw (CLK, CEB, WEB, A, D, WMASK, Q);

    //-------------------------------------------------------------------------
    // Parameters — 사용자가 instantiate할 때 override 가능.
    //
    // 예시: sram_1rw #(.DATA_WIDTH(64), .DEPTH(16384)) u_buf (...);
    //   --> 64-bit wide, 16384 entries deep -> 총 128 KB
    //-------------------------------------------------------------------------

    parameter DATA_WIDTH = 32;      // Word width (bit 단위)
    parameter DEPTH      = 1024;    // Word 개수 (memory row 수)
    parameter PIPELINE   = 0;       // 0 = 1-cycle read latency
                                    // 1 = 2-cycle read latency (큰 macro 매칭용)
    parameter INIT_FILE  = "";      // Sim 전용: memory에 preload할 hex 파일.
                                    // 빈 string이면 비활성. Synthesis는 무시.

    //-------------------------------------------------------------------------
    // clog2 함수 — "ceiling log base 2".
    //
    // `value` 개수의 entry를 address로 표현하려면 몇 bit가 필요한지 반환.
    //   clog2(1)    = 0   (1 entry는 0 bit로 표현 가능 — 단 아래에선 1로 처리)
    //   clog2(2)    = 1   (0..1 → 1 bit)
    //   clog2(8)    = 3   (0..7 → 3 bit)
    //   clog2(1024) = 10  (0..1023 → 10 bit)
    //   clog2(1000) = 10  (1000개도 10 bit 필요 — 올림 처리)
    //
    // 왜 "c"log2? 수학의 log2(1000) = 9.97 → 정수로 표현하려면 올림(ceiling)
    // 해서 10. 그래서 ceiling-log-2.
    //
    // 알고리즘: (value - 1)을 0이 될 때까지 오른쪽 shift하면서 횟수를 센다.
    // 그 횟수가 곧 필요한 bit 수.
    //
    // 왜 직접 함수로 정의? SystemVerilog에는 $clog2 built-in이 있지만
    // (V2005+), strict Verilog-2001에는 없음. 우리가 직접 만들면 어떤
    // tool에서도 portable.
    //-------------------------------------------------------------------------
    function integer clog2;
        input integer value;
        integer i;
        begin
            clog2 = 0;
            for (i = value - 1; i > 0; i = i >> 1)
                clog2 = clog2 + 1;
        end
    endfunction

    // ADDR_WIDTH는 DEPTH로부터 자동 계산. 사용자가 override 불가능
    // (parameter가 아니라 localparam이므로).
    localparam ADDR_WIDTH = clog2(DEPTH);

    //-------------------------------------------------------------------------
    // Port 방향 및 width 선언 (non-ANSI 스타일 연속).
    //
    // Polarity convention은 foundry memory-compiler 관습을 따름:
    //   - CEB / WEB은 "bar" signal = active-LOW.
    //   - WMASK는 active-HIGH (1 = "이 bit 써라").
    //
    // 왜 enable이 active-low? 산업 전통: 초기화 안 된 상태(uninitialized = 1)
    // 가 자동으로 SAFE state(chip disabled, 아무것도 안 써짐)가 되도록.
    //-------------------------------------------------------------------------
    input                       CLK;     // Clock. 모든 input은 rising edge에서 sampling.
    input                       CEB;     // Chip Enable Bar (active-low).
                                         //   0 = access (read 또는 write)
                                         //   1 = idle (mem이든 Q든 아무것도 변하지 않음)
    input                       WEB;     // Write Enable Bar (active-low).
                                         //   0 = write (단, CEB도 0일 때)
                                         //   1 = read  (단, CEB도 0일 때)
    input  [ADDR_WIDTH-1:0]     A;       // Word address.
    input  [DATA_WIDTH-1:0]     D;       // Write data (write cycle에서만 의미 있음).
    input  [DATA_WIDTH-1:0]     WMASK;   // Bit-write enable (active-HIGH).
                                         //   WMASK[i]=1 -> i번째 bit 쓰기
                                         //   WMASK[i]=0 -> i번째 bit 유지
    output [DATA_WIDTH-1:0]     Q;       // Read data. Latency는 PIPELINE에 따라 달라짐.

    //-------------------------------------------------------------------------
    // 내부 storage.
    //
    // `reg [DATA_WIDTH-1:0] mem [0:DEPTH-1]`는 Verilog의 memory 선언 문법:
    // `DEPTH`개의 register array, 각 register는 `DATA_WIDTH` bit width.
    //
    // Synthesis 시 어떻게 처리되는가:
    //   (a) 아주 작은 DEPTH(예: 32)면 register-based RAM으로 추론
    //   (b) FPGA에서는 block RAM 추론
    //   (c) ASIC PNR에서는 foundry SRAM macro로 **통째로 교체**
    // 우리가 노리는 건 (c) 케이스.
    //
    // Q_pre는 registered read-data output. 아래에서 PIPELINE 값에 따라
    // Q를 Q_pre에 그대로 연결하거나 (PIPELINE=0), 한 단계 더 register를
    // 거치게 한다 (PIPELINE=1).
    //-------------------------------------------------------------------------
    reg [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    reg [DATA_WIDTH-1:0] Q_pre;

    //-------------------------------------------------------------------------
    // Sim-only memory 초기화.
    //
    // `initial` block은 simulation 시작 시 단 한 번만 실행됨. Synthesis는
    // 이걸 무시 (실제 SRAM은 power-on 시 값이 RANDOM이라 default를 만들 수
    // 없음). `translate_off/on` pragma는 synthesis tool에게 "이 block은
    // 절대 합성하지 마라" 하고 belt-and-suspenders로 명시.
    //
    // `$readmemh`는 hex 값들의 text file(한 줄에 하나씩)을 mem array에
    // address 0부터 차례로 읽어 들임. Testbench에서 weight, lookup table,
    // golden vector 등을 preload할 때 유용.
    //
    // INIT_FILE이 "" (default)이면 아무것도 안 일어남.
    //-------------------------------------------------------------------------
    // synthesis translate_off
    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end
    // synthesis translate_on

    //-------------------------------------------------------------------------
    // Write logic — bit-wise mask 적용.
    //
    // Trigger: CLK rising edge에서, CEB=0 AND WEB=0일 때만 (valid write).
    //
    // Masked-write 공식:
    //
    //   new = (old & ~mask) | (new_data & mask)
    //         ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^
    //         이 bit들은 유지   이 bit들은 덮어씀
    //
    // 각 bit 위치 i에 대해:
    //   - mask[i]=1: -> ~mask[i]=0이라 old 항은 0; D[i]가 흘러옴. 결과 = D[i].
    //   - mask[i]=0: -> ~mask[i]=1이라 old[i] 유지; D[i]는 차단. 결과 = old[i].
    //
    // 예시 (DATA_WIDTH=8):
    //   mem[A]   = 8'b11111111  (현재 값)
    //   D        = 8'b00000000  (쓰려는 값)
    //   WMASK    = 8'b00001111  (하위 4 bit만 쓰기)
    //   결과     = 8'b11110000  (상위 nibble 유지, 하위 nibble만 clear)
    //
    // 실제 foundry macro는 이걸 "BWEB" (Bit Write Enable Bar, active-low)
    // 형태로 노출 — 우리는 active-high WMASK를 쓰고 wrapper에서 invert.
    //-------------------------------------------------------------------------
    always @(posedge CLK) begin
        if (!CEB && !WEB) begin
            mem[A] <= (mem[A] & ~WMASK) | (D & WMASK);
        end
    end

    //-------------------------------------------------------------------------
    // Read logic.
    //
    // Trigger: CLK rising edge에서, CEB=0 AND WEB=1일 때만 (read access).
    // mem[A]를 Q_pre(registered output stage)에 capture.
    //
    // 핵심: Q_pre는 valid read일 때만 update됨. Write나 idle cycle에는
    // Q_pre가 이전 값을 HOLD — 이게 실제 macro 동작과 일치한다 (write 중
    // 이나 idle 중에는 output이 변하지 않음).
    //
    // Write logic과 SEPARATE된 always block임에 주의. 합쳐도 합성 결과는
    // 같지만, "이 block은 mem에 쓴다, 이 block은 mem을 읽는다"로 분리하면
    // 의도가 더 명확.
    //-------------------------------------------------------------------------
    always @(posedge CLK) begin
        if (!CEB && WEB) begin
            Q_pre <= mem[A];
        end
    end

    //-------------------------------------------------------------------------
    // 선택적 output pipeline register (PIPELINE parameter).
    //
    // `generate`는 compile-time 구문. PIPELINE 값에 따라 두 갈래 중 하나만
    // 합성됨:
    //
    //   PIPELINE=0:  Q는 Q_pre 그 자체.
    //                read latency = 1 cycle (cycle N에 read 명령 → N+1에 Q valid)
    //
    //   PIPELINE=1:  Q는 Q_reg를 거침 — Q_pre 다음에 flip-flop 하나 더.
    //                read latency = 2 cycle (cycle N에 read 명령 → N+2에 Q valid)
    //
    // 왜 macro swap에서 이게 중요한가:
    //   - 작은 foundry macro (예: 4 KB)는 보통 1-cycle latency.
    //   - 큰 foundry macro (예: 256 KB+)는 보통 2-cycle latency. 이유는
    //     내부 bitline / sense-amp delay 때문에 한 cycle에 다 못 끝나서
    //     output stage register가 강제됨.
    //   - 같은 RTL로 PIPELINE 값만 바꾸면 양쪽 다 cover 가능.
    //
    // Q를 받아 쓰는 쪽은 latency를 인지해야 함. Testbench는 PIPELINE을
    // 보고 적절한 cycle 수만큼 기다린 다음 Q를 check한다.
    //
    // `begin : name`은 named generate block. Verilog가 multi-branch
    // generate에서 이름을 요구한다 (g_no_pipe / g_pipe).
    //-------------------------------------------------------------------------
    generate
        if (PIPELINE == 0) begin : g_no_pipe
            // 1-cycle latency 경로: combinationally Q를 Q_pre에 직결.
            assign Q = Q_pre;
        end else begin : g_pipe
            // 2-cycle latency 경로: flip-flop 한 단계 추가.
            reg [DATA_WIDTH-1:0] Q_reg;
            always @(posedge CLK) begin
                Q_reg <= Q_pre;
            end
            assign Q = Q_reg;
        end
    endgenerate

endmodule
