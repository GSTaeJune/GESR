# MXP TOP Driving Sequence — Task 7 Reference

본 문서는 `gemm_sram` 통합 TB (`tb/gemm_sram_top_tb.v`) 의 Task 7 **DRIVE** 단계
작성용 cheat-sheet. cycle-by-cycle 발사 패턴은 이미 다음 두 곳에서 한 번 완전히
정리됐다 — 본 문서는 그쪽을 가리키고, **gemm_sram TB 관점** 의 빠진 조각만
채운다.

## Authoritative sources (먼저 이걸 읽는다)

1. **`precision_modes_protocol.md`** (repo root, 299 lines)
   - v1.0, 2026-05-11 validated. 9 모드 (A∈{8,4,2} × W∈{8,4,2}) full GEMM 128³ PASS.
   - 이 문서 안의 모든 cycle 수치, chain entry, fire delay, scale 공식, 파일명
     convention 은 실측 검증된 값이다. **Task 7 가 답습할 sequencing 본체.**
2. **`CLAUDE.md` § "MXP control surface"** (repo root)
   - 5 chain 의 entry / propagation 방향 한 페이지 요약. 빠르게 reload 할 때만 사용.

본 문서는 위 두 문서를 **재서술하지 않는다**. 섹션 번호로 가리킨다.

---

## 1. 입력 파일 — TB 가 어떻게 `$readmemh` 로 받는가

파일명 규칙은 `precision_modes_protocol.md` §4 표 참조 (예: `a_input_BS_mxint8.hex`,
`b_input_mxint8.hex`, `a_scale_mxint8.hex`, `b_scale_mxint8.hex`,
`c_mxint8_mxint8.hex`).

**현재 통합 plan 의 경로 convention** (`docs/superpowers/specs/2026-05-14-integration-design.md`
와 `plans/2026-05-14-integration-implementation.md` Task 7 step 2 에 박힌 값):

```verilog
// WORK_DIR plusarg, default "work/A8_B8" (또는 mode 별 디렉토리)
$readmemh({WORK_DIR, "/hw_input/a_input_BS_8.hex"}, a_bs);
$readmemh({WORK_DIR, "/hw_input/b_input_8.hex"},    b_bp);
$readmemh({WORK_DIR, "/hw_input/a_scale_8.hex"},    a_scale);
$readmemh({WORK_DIR, "/hw_input/b_scale_8.hex"},    b_scale);
```

**Note**: 현재 plan 의 파일명 (`a_input_BS_{P}.hex`, `b_input_{P}.hex`, …) 은
protocol doc 의 `_mxint{P}` suffix 와 다르다. Task 7 는 실제 `MXP_Tools` /
`TransformSerial.py` 출력의 파일명에 맞춰야 한다 — sim 돌리면서 첫 시도에
`$readmemh: cannot open file` 나오면 경로 확정 후 plan 수정.

### Reg 배열 sizing (128×128×128 GEMM 기준, M_T = K_T = N_T = 4)

| reg 배열 | 사이즈 (depth × width) | 출처 / 의미 |
|---|---|---|
| `a_bs`     | `K_T × M_T × TILE_SIZE × W_CYC` × 32 bit | weight bit-serial. row 당 W bit. A8W8 = 4·4·32·8 = 4096 lines |
| `b_bp`     | `N_T × K_T × TILE_SIZE` × 256 bit         | activation byte-parallel 256-bit / col-tile. A8W8 = 4·4·32 = 512 lines |
| `a_scale`  | `K_T × M_T × TILE_SIZE` × 8 bit           | E8M0 weight scale. A8W8 = 4·4·32 = 512 lines |
| `b_scale`  | `N_T × K_T × TILE_SIZE` × 32 bit (= 4×8)  | E8M0 activation scale, 4-lane packed per col. A8W8 = 512 lines |
| `c_ref`    | `M × N` × 32 bit signed                   | INT MAC golden. 128·128 = 16384 lines |

(W_CYC, N_T_logical 은 모드별 — `precision_modes_protocol.md` §1 Mode Matrix 참조.
A_INT4 / A_INT2 에서는 `b_bp` 의 의미가 바뀐다: lane packing 됨 — §2 참조.)

---

## 2. Cycle-by-cycle DRIVE — 새로 쓰지 말 것

전체 sequencing 은 `precision_modes_protocol.md` §3 "Driver Protocol — 5 Input Chains"
에서 끝까지 정의됨. 요점만 인용:

- **Stage 0 Reset**: §3 Stage 0
- **Stage 2-A Initial B load (32 cy)**: §3 Stage 2-A — `in_b`, `in_Scale_Activation`,
  `in_Station_control`, `in_loadEN=32'hFFFFFFFF`, `in_station_loadEN=1`,
  `in_station_control=0` (Buf1 target).
- **Stage 2-B Settle + Buf1 toggle (~32 cy)**: §3 Stage 2-B — 16 idle + 1 toggle +
  14 settle. Settle ≥14 cy 가 symmetric chain max delay (15) 보다 짧으면 broadcast 가
  안 맞아서 fire 가 dead.
- **Stage 3+4 MAC sweep**: §3 Stage 3+4 — `o = 0..TILE_SIZE×W_CYC-1` inner loop.
  Per-cycle drive 6 가지 (weight bit / PE in_control toggle / Station toggle /
  start_accumulate / Wcontrol / scale_weight) 의 정확한 cycle 조건은 §3 본문 표.
- **Stage 5 Tail (60+ cy)**: §3 Stage 5.

**Mode-specific 상수** (`TOGGLE`, `FIRST_FIRE`, `A_fire_delay`, `FIRES_PER_COL`,
`W_CYC`) 전부 `precision_modes_protocol.md` §1 Mode Matrix + §3 본문 에 실측치 박혀있다.
Task 7 는 그 표에서 (A_PREC, B_PREC) → 값 lookup 해서 코드에 박는 방식.

---

## 3. Fire timing — per-column skew (Task 7 RMW scheduler 가 봐야 함)

`out_fire` 는 **per-col**. 두 종류의 skew 가 발생:

### 3.1 Chain entry skew (단방향 station data chain)
- station 데이터 chain (in_b 와 동행) 은 col 31 entry, leftward propagation.
- col c 의 in_b / station data 도착 시점 = `TB_drive_cyc + (31 - c)` cycle (delay
  per col = `31 - c`, max 31 at col 0).
- 따라서 같은 K-tile 의 first fire 는 col 31 이 먼저, col 0 이 31 cycle 늦게.

### 3.2 Symmetric chain skew (acc chain + station selector chain)
- acc chain (`in_start_accumulate`, `in_Wcontrol`, `in_scale_weight`) 과 station
  selector chain (`in_station_control`) 은 col 16 entry, symmetric outward fan.
- col c 도착 delay = `|c - 16|` cycle (max 15 at col 0 or 31, 0 at col 16).

### 3.3 종합 — col c 의 first fire cycle
`precision_modes_protocol.md` §3 "FIRST_FIRE formula":
```
FIRST_FIRE_global = 17 + 1 + W_CYC + A_fire_delay      // (col 16 기준)
```

col c 에서는 acc chain delay `|c-16|` 추가되어:
```
FIRST_FIRE_col(c) ≈ FIRST_FIRE_global + |c - 16|
```

| 모드  | FIRST_FIRE (col 16) | FIRST_FIRE (col 0/31) |
|---|---|---|
| A8W8 | 28 | 43 |
| A8W4 | 24 | 39 |
| A8W2 | 22 | 37 |
| A4W8 | 27 | 42 |
| A4W4 | 23 | 38 |
| A4W2 | 21 | 36 |
| A2W8 | 26 | 41 |
| A2W4 | 22 | 37 |
| A2W2 | 20 | 35 |

(±보정: in_b chain 의 leftward propagation 과 acc chain 의 symmetric propagation 이
일치하도록 protocol doc 가 chain topology 를 그렇게 설계함 — col 별로 발사 cycle 다르게
줄 필요 **없다**, scalar entry 한 번이면 chain delay 가 자동으로 fire window 정렬한다.
TB 의 책임은 entry signal 을 §3 cycle 에 맞춰 한 번 발사하는 것뿐.)

### 3.4 후속 fire 간격
한 K-tile 안에서 한 col 은 `FIRES_PER_COL / (K_T × N_T)` 번 fire — fire 간격은
`W_CYC` cycle (Accumulator 의 cnt rollover). 자세히는 §1 Mode Matrix `W_CYC` 열
+ §3 Stage 3+4 "scale_weight drive" 항목.

---

## 4. gemm_sram TB 가 추가로 신경써야 할 것

protocol doc 은 MXP standalone 기준이다. 통합 TB 는 다음이 더 붙는다:

1. **RMW dispatch.** `out_fire[c]` rising 감지 → col c 의 `out_accumulate` lane
   슬라이스 (mode-aware, `precision_modes_protocol.md` §2 layout) → `rmw_in_GEMM`
   port 로 발사. lane→C 매핑은 Task 6 산출물 (`lane-to-c-mapping.md`) 참조.
2. **RMW latency 와 SRAM read-issue 정렬.** RMW 내부 latency = `L_CONV + L_ADD = 5`
   cycle (`gemm_sram.srcs/sources_1/new/RMW.v` 헤더). SRAM read 는 PIPELINE 설정에
   따라 1~2 cy. read addr 는 fire 보다 `L_CONV` cy 앞서 issue 해야 add 시점에 도착.
3. **First-tile init.** K-tile 0 의 첫 RMW 는 SRAM 에 garbage FP32 (NaN 가능) 가
   있으면 깨진다. Task 2 의 zero-priming loop 가 `tb/gemm_sram_top_tb.v` L128
   에 이미 있음 — DRIVE 이전에 반드시 돌아야 함.
4. **Bank collision.** 32 col 이 16 bank 에 매핑됨 (interleaved, Task 3).
   같은 cycle 에 2 col 의 fire 가 같은 bank 로 가면 충돌. col 별 fire skew (§3.3) 가
   자연스럽게 분산시키지만, RMW 의 5-cy latency 가 끼면 collision window 가 늘어남.
   Task 7 에서 sim waveform 보며 collision 발생 시 per-col FIFO 추가 결정.
5. **Mode boundary alignment.** A_ctrl / W_ctrl 변경은 K-tile boundary 에서만 (mid-K
   변경 = partial sum 깨짐). `precision_modes_protocol.md` §6.2 #5.

---

## 5. Open uncertainties (Task 7 에서 sim 으로 확정할 것)

- **파일명 suffix**: plan 의 `_{P}` vs protocol doc 의 `_mxint{P}` — 실제 MXP_Tools
  output 파일명 확인 후 통일.
- **TOGGLE 값 검증**: protocol doc §3 Stage 3+4 #3 의 A8W4 / A4W8 TOGGLE 은 "TBD
  verify". A8W2 / A4W4 / A4W2 / A2*  는 실측 PASS. Task 7 A8W8 → A8W4 확장 시 sim
  돌려서 TOGGLE 정확값 확정.
- **scale_weight 인덱싱**: protocol doc §3 #6 의 `m_idx % (K_T × M_T × TILE_SIZE)`
  modulo — A8W8 에서 PASS 검증됐지만 N_T_logical > 1 (A4 / A2) 에서 wrap behavior
  재확인 필요.

---

## 6. 한 줄 요약

> Task 7 implementer 는 `precision_modes_protocol.md` §1 + §3 을 옆에 띄워놓고
> mode 별 상수 (W_CYC, A_fire_delay, FIRST_FIRE, TOGGLE) 만 코드에 박으면 됨.
> 본 문서는 그 protocol doc 이 다루지 않는 **통합 TB 관점** (파일 로딩, RMW
> dispatch 정렬, bank collision) 만 추가로 정리했다.
