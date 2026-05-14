# A4/A2 Mode Lane → C[m,n] Mapping — Task 6 Reference

본 문서는 통합 TB (Task 7+) 에서 RMW 발사 시 어느 lane 의 결과를 어느 SRAM
주소에 write 할지 결정하는 근거. **`Accumulator_Col.out_accumulate[59:0]` 의 mode
별 lane 패킹** 이 SW reference (`mxint_gemm_golden`) 의 `C[m, n]` 어느 위치를
담당하는지 dispatch table 로 정리한다.

## Authoritative sources (먼저 이걸 본다)

1. **`precision_modes_protocol.md` §2 "Lane → N-Position Packing"** (repo root)
   - 실측 검증된 lane→N 매핑 본체. 이 문서는 §2 를 RMW dispatch 관점으로
     **재해석** 한 것 — 새로운 사실 추가는 없음.
2. **`gemm_sram.srcs/sources_1/imports/Desktop/MXP/MXP.srcs/sources_1/new/Accumulator_Col.v`**
   - L196–260 mode mux: `out_accumulate` / `out_scale` 비트 패킹 RTL.
   - L129–132 `comb_s0..s3`: 9-bit signed scale sub-word 4 개 정의.
3. **`MXP_Tools/mxp_tools/gemm.py::mxint_gemm_golden`** (L25–75)
   - SW reference. **lane 패킹을 모름** — 그냥 `(M, K) @ (K, N)` numpy matmul.
     `C[m, n]` 은 row-major `(M, N)` FP32 행렬. lane 매핑은 SW reference 의
     특성이 **아니라** HW 가 32-col SA 로 N=128 을 커버하기 위한 packing 의
     역연산이다. 따라서 본 문서의 "lane → n" 은 RTL packing 의 inverse.
4. **`docs/superpowers/notes/mxp-driving-sequence.md`** §3 — col j 의 fire timing
   skew.

---

## 1. 공통: m-차원 (row) 매핑은 mode 와 무관

세 mode 모두 `m` 차원은 **SA row (m_in ∈ 0..31)** 가 담당하고, m-tile 인덱스
`m_t` 는 K-tile 내부 inner loop 순서로 결정된다. 즉:

```
m_global = m_t × TILE_SIZE + m_in     (m_in ∈ 0..31, m_t ∈ 0..M_T-1)
```

`m_in` 은 fire 카운터 (`fire_cnt[c] % TILE_SIZE`) 로 추적. mode 별 lane 분기는
**n 차원 매핑에만 영향** — m 매핑은 동일.

## 2. col j 와 n-tile (n_t) — mode 별 의미가 다르다

`precision_modes_protocol.md` §1 + §2 요약:

| A_ctrl  | `N_T_logical` | 1 station byte 당 N-position 수 | col j 가 담당하는 n positions per fire |
|---|---|---|---|
| A_INT8  | 4 | 1 | n = `n_t × 32 + j`                                   (1 개) |
| A_INT4  | 2 | 2 | n ∈ {`n_pair × 64 + 32 + j`, `n_pair × 64 + j`}      (2 개) |
| A_INT2  | 1 | 4 | n ∈ {`96 + j`, `64 + j`, `32 + j`, `0 + j`}          (4 개) |

(N=128 가정. M=K=N=128, M_T=K_T=4, TILE_SIZE=32 이라는 spec L43 표 기준.
A_INT2 는 N_T_logical=1 이므로 `n_t` loop 가 한 번만 돈다.)

`n_t` 는 driving sequence 의 outermost loop (`n_t → k_t → m_t → o`). col j 의
fire 가 어느 `n_t` 의 결과인지는 `fire_cnt[c]` / (`K_T × M_T × TILE_SIZE`) 의
몫으로 추적 (자세히는 `mxp-driving-sequence.md` §3.4).

---

## 3. A_INT8 모드 (1 lane per col)

### 비트 layout (`out_accumulate[60c +: 60]`)
- `[60c + 20 : 60c + 0]` = `s2[20:0]` — 21-bit signed INT MAC.
- 상위 39비트는 sign-ext (`Accumulator_Col.v` L249).

### Scale (`out_scale[36c +: 36]`)
- `[36c + 8 : 36c + 0]` = `comb_s0` (lower 9-bit) — `Accumulator_Col.v` L251.
- 상위 27비트는 0.

### Lane → C[m, n]
| lane idx | acc slice | scale slice | n offset |
|---|---|---|---|
| 0 | `out_accumulate[60c + 20 : 60c + 0]` | `out_scale[36c + 8 : 36c + 0]` | `n_t × 32 + j` |

### Dispatch
- col j fire 1 회 당 **RMW 1 회** 발사.
- C 주소: `(m_global, n_t × 32 + j)`.
- 발사 총수 = `M × N = 128 × 128 = 16384` (전체 GEMM 기준).

---

## 4. A_INT4 모드 (2 lanes per col)

### 비트 layout (`out_accumulate[60c +: 60]`, `Accumulator_Col.v` L241–242)
- `[60c + 35 : 60c + 18]` = `s1_a[17:0]` — **top nibble** path
  (lane0+lane1 의 `(lane0 << 2) + lane1`).
- `[60c + 17 : 60c +  0]` = `s1_b[17:0]` — **bot nibble** path
  (lane2+lane3 의 `(lane2 << 2) + lane3`).
- `[60c + 59 : 60c + 36]` = 0 (zero-pad).

### Scale (`out_scale[36c +: 36]`, `Accumulator_Col.v` L243)
- `[36c + 17 : 36c +  9]` = `comb_s0` — top (n_t input `2*n_pair + 1` 의 scale).
- `[36c +  8 : 36c +  0]` = `comb_s1` — bot (n_t input `2*n_pair + 0` 의 scale).
- `[36c + 35 : 36c + 18]` = 0.

### Lane → C[m, n]
`precision_modes_protocol.md` §2 A_INT4 직접 인용. 활성화 file 의 `n_t` 인덱스가
"top = `2*n_pair + 1`, bot = `2*n_pair + 0`" 이므로 (b_input_mxint4 의 file 내
n_t 순서를 반대로 packing 함):

| lane | acc slice                              | scale slice                            | n offset (per fire)        |
|---|---|---|---|
| top (s1_a) | `out_accumulate[60c + 35 : 60c + 18]` | `out_scale[36c + 17 : 36c + 9]` | `n_pair × 64 + 32 + j` |
| bot (s1_b) | `out_accumulate[60c + 17 : 60c +  0]` | `out_scale[36c +  8 : 36c + 0]` | `n_pair × 64 +  0 + j` |

여기서 `n_pair` = N-tile pair index. `N_T_logical=2` 이므로 `n_pair ∈ {0, 1}`
(A4 에서 N=128 커버하려면 2 회 pair sweep 필요). `fire_cnt[c]` 의 `K_T × M_T ×
TILE_SIZE = 512` 주기마다 `n_pair` 가 증가.

### Dispatch
- col j fire 1 회 당 **RMW 2 회** 발사 (top + bot, n offset 32 차이).
- C 주소 2 개: `(m_global, n_pair*64 + 32 + j)` 와 `(m_global, n_pair*64 + j)`.
- 두 RMW 가 같은 cycle 에 같은 m / 다른 n → bank collision 발생 시 어떻게 직렬화
  할지는 Task 7 sim 에서 결정 (`mxp-driving-sequence.md` §4.4).

---

## 5. A_INT2 모드 (4 lanes per col)

### 비트 layout (`out_accumulate[60c +: 60]`, `Accumulator_Col.v` L234)
- `[60c + 59 : 60c + 45]` = `out_INT2_0` — 15-bit signed (acc_len).
- `[60c + 44 : 60c + 30]` = `out_INT2_1`.
- `[60c + 29 : 60c + 15]` = `out_INT2_2`.
- `[60c + 14 : 60c +  0]` = `out_INT2_3`.

### Scale (`out_scale[36c +: 36]`, `Accumulator_Col.v` L235)
- `[36c + 35 : 36c + 27]` = `comb_s0` (lane0).
- `[36c + 26 : 36c + 18]` = `comb_s1` (lane1).
- `[36c + 17 : 36c +  9]` = `comb_s2` (lane2).
- `[36c +  8 : 36c +  0]` = `comb_s3` (lane3).

### Lane → C[m, n]
`precision_modes_protocol.md` §2 A_INT2 직접 인용. station byte 의 비트 위치와
lane 인덱스 정렬:

| lane | acc slice                              | scale slice                            | n offset |
|---|---|---|---|
| 0 (out_INT2_0) | `out_accumulate[60c + 59 : 60c + 45]` | `out_scale[36c + 35 : 36c + 27]` | `96 + j` |
| 1 (out_INT2_1) | `out_accumulate[60c + 44 : 60c + 30]` | `out_scale[36c + 26 : 36c + 18]` | `64 + j` |
| 2 (out_INT2_2) | `out_accumulate[60c + 29 : 60c + 15]` | `out_scale[36c + 17 : 36c +  9]` | `32 + j` |
| 3 (out_INT2_3) | `out_accumulate[60c + 14 : 60c +  0]` | `out_scale[36c +  8 : 36c +  0]` | ` 0 + j` |

A_INT2 는 `N_T_logical = 1` → `n_t` loop 없음. 한 fire 가 32-col × 4-lane = 128
positions = N 전체 를 cover.

### Dispatch
- col j fire 1 회 당 **RMW 4 회** 발사 (lane0..3, n offset 0/32/64/96).
- C 주소 4 개: `(m_global, lane*32_inverted + j)`. 위 표의 96/64/32/0 stride 는
  lane0 가 가장 높은 n, lane3 가 가장 낮은 n 임을 명시.
- 동시 4 RMW → bank collision 가능성 가장 높음. Task 7 sim 에서 직렬화 방식
  (FIFO depth / per-lane RMW unit 개수) 결정.

---

## 6. Dispatch 요약 (Task 7 RMW scheduler 용)

| A_ctrl  | RMW per col fire | acc width per lane | scale per lane | n positions per fire (col j) |
|---|---|---|---|---|
| A_INT8  | 1 | 21-bit signed | comb_s0 (9-bit signed) | `{n_t × 32 + j}` |
| A_INT4  | 2 | 18-bit signed | comb_s0, comb_s1       | `{n_pair × 64 + 32 + j, n_pair × 64 + j}` |
| A_INT2  | 4 | 15-bit signed | comb_s0..s3            | `{96+j, 64+j, 32+j, 0+j}` |

**Total RMW per GEMM (M=K=N=128, M_T=K_T=4, TILE_SIZE=32):**

```
RMW_total = M × N × K_T = 128 × 128 × 4 = 65536        (mode 무관 — 모든 K-block 마다 RMW)
```

mode 별 col 당 fire 수는 다르지만 (`FIRES_PER_COL`: A8=2048, A4=1024, A2=512),
1 fire 당 RMW 수가 1/2/4 이므로 곱은 동일.

| Mode  | FIRES_PER_COL | RMW per fire | RMW per col | × 32 col = total |
|---|---|---|---|---|
| A8 *  | 2048 | 1 | 2048 | **65536** |
| A4 *  | 1024 | 2 | 2048 | **65536** |
| A2 *  |  512 | 4 | 2048 | **65536** |

(`precision_modes_protocol.md` §1 Mode Matrix 의 FIRES_PER_COL 컬럼.)

---

## 7. RMW dispatcher RTL 의 mode-aware 슬라이싱 (참고 skeleton)

```verilog
// 가정: col c 의 fire 가 raise, mode_oh = {is_A2, is_A4, is_A8}
// `acc` = out_accumulate[60*c +: 60], `sc` = out_scale[36*c +: 36]
case (mode_oh)
  3'b001: begin // A8 — 1 lane
    rmw_in_GEMM[0] <= {{11{acc[20]}}, acc[20:0]};   // 21b signed → 32b sign-ext
    rmw_scale[0]   <= sc[8:0];                       // comb_s0
    rmw_n_off[0]   <= n_t * 32 + c;
    rmw_valid      <= 4'b0001;
  end
  3'b010: begin // A4 — 2 lanes
    rmw_in_GEMM[0] <= {{14{acc[35]}}, acc[35:18]};  // top (s1_a) 18b signed
    rmw_in_GEMM[1] <= {{14{acc[17]}}, acc[17: 0]};  // bot (s1_b) 18b signed
    rmw_scale[0]   <= sc[17:9];                      // comb_s0
    rmw_scale[1]   <= sc[ 8:0];                      // comb_s1
    rmw_n_off[0]   <= n_pair * 64 + 32 + c;
    rmw_n_off[1]   <= n_pair * 64 +  0 + c;
    rmw_valid      <= 4'b0011;
  end
  3'b100: begin // A2 — 4 lanes
    rmw_in_GEMM[0] <= {{17{acc[59]}}, acc[59:45]};  // 15b signed
    rmw_in_GEMM[1] <= {{17{acc[44]}}, acc[44:30]};
    rmw_in_GEMM[2] <= {{17{acc[29]}}, acc[29:15]};
    rmw_in_GEMM[3] <= {{17{acc[14]}}, acc[14: 0]};
    rmw_scale[0]   <= sc[35:27];                     // comb_s0
    rmw_scale[1]   <= sc[26:18];                     // comb_s1
    rmw_scale[2]   <= sc[17: 9];                     // comb_s2
    rmw_scale[3]   <= sc[ 8: 0];                     // comb_s3
    rmw_n_off[0]   <= 96 + c;
    rmw_n_off[1]   <= 64 + c;
    rmw_n_off[2]   <= 32 + c;
    rmw_n_off[3]   <=  0 + c;
    rmw_valid      <= 4'b1111;
  end
  default: rmw_valid <= 4'b0000;  // IDLE
endcase
```

(실제 Task 7 구현에서는 4 개 RMW unit 직렬화 / FIFO 방식 결정 후 위 skeleton
조정. acc width sign-extension 폭은 RMW 의 `in_GEMM[31:0]` INT32 input 기준.)

---

## 8. TBD — Task 7 sim 에서 확정할 것

다음 항목은 정적 RTL 분석만으로는 100% 확실하지 않으니 sim waveform 으로
재확인:

1. **A4 의 `n_pair` 인덱싱 wrap timing.** `mxp-driving-sequence.md` §5 의
   "scale_weight 인덱싱" TBD 와 동일 — N_T_logical > 1 에서 wrap behavior 가
   protocol doc 의 의도대로 도는지 sim 확인.
2. **`n_pair` 와 fire_cnt 관계의 정확한 모듈로**. A4 의 `FIRES_PER_COL = 1024` 이고
   `K_T × M_T × TILE_SIZE = 512` → `n_pair = fire_cnt[c] / 512`. **A4 에서만**
   `n_pair ∈ {0, 1}` 로 wrap 한다는 것이 RTL 적으로 명확하나, MXP_Tools 의
   activation file `b_input_mxint4.hex` 의 실제 n_t 인덱싱 순서 (file 의
   `2*n_pair + 1` 이 top 인지 bot 인지) 는 sim 에서 한 번 waveform 으로 확인 후
   못 박을 것. protocol doc §2 의 "n_t = `2*n_pair + 1` = top" 명시를 일단 truth
   로 가정.
3. **A2 lane 순서 (96/64/32/0 stride)**. station byte 의 비트 `[7:6]=lane0` →
   `comb_s0` → `N = 96 + c` 매핑은 protocol doc §2 와 `Accumulator_Col.v` L234
   둘 다에서 일관. 다만 `b_input_mxint2.hex` 의 n_t = 3/2/1/0 → lane 0/1/2/3 가
   sim 에서 그대로 나오는지는 한 번 waveform 으로 검증.
4. **K-tile 누적 시점**. RMW dispatch 는 col 마다 fire 마다 발사되나, 동일
   `(m_global, n)` 에 대해 K-block 4 회 (K_T=4) 누적해야 최종 C[m,n] 가
   완성된다. RMW 가 매 K-block 마다 SRAM read-update-write 를 하는 구조이므로
   K-tile 진행 순서 = SRAM write 의 순차 update 순서. 본 문서의 매핑은 "단일
   K-block 의 1 fire" 기준이며, K-tile 누적은 RMW dispatcher 의 책임이지
   lane→C 매핑의 책임이 아님.

위 항목은 매핑 자체의 ambiguity 가 아니라 **driving sequence 와 file 인덱싱의
교차 검증** 이다. lane → n offset 의 핵심 수치 (32+j / j, 96+j / 64+j / 32+j /
j) 는 protocol doc §2 와 `Accumulator_Col.v` 두 곳에서 일관되므로 Task 7
RMW dispatcher 의 base table 로 그대로 박아도 안전.
