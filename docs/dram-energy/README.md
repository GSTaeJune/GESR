# DRAM 에너지 계수 (`dram_presets.json`) — 출처 · 도출 · 선택

MXP_scheduler 의 `coeffs.dram` (pJ/bit) 값을 **어떤 경계로, 어떤 1차 출처에서, 어떻게** 도출했는지 기록한다.
스케줄러가 DRAM 트래픽 1 bit 당 매기는 에너지 비용이라 랭킹의 dram-vs-onchip 트레이드오프를 직접 좌우한다.

확정일: 2026-06-23 · 사용자 결정 2건: **주력 = LPDDR5/5X (edge/mobile)**, **경계 = full-system 통일**.

---

## 1. 에너지 경계 (boundary) — full-system 으로 통일

`pj_per_bit` 은 DRAM 1 bit 전송에 시스템이 실제로 지불하는 **전체** 에너지로 통일한다:

```
device core (activate/row + read/column + background)
  + on-die I/O (read driver + termination)
  + SoC PHY + memory controller
  + refresh
```

> **왜 통일이 필요했나**: 직전(2026-06-10) 시드 테이블은 경계가 섞여 있었다 — LPDDR 계열은 device-internal(~5 pJ/b),
> DDR 계열은 off-chip I/O 포함(~15 pJ/b). 한 칩 안의 매핑 랭킹엔 무관하지만 계열 간 비교·절대 에너지 신뢰도가 깨진다.
> full-system 은 (a) on-chip SRAM 대비 DRAM 비용을 과소평가하지 않고, (b) 모든 프리셋을 같은 기준에 올린다.

경계에 따라 값이 2~3배 차이 난다. 같은 LPDDR4라도 device-core ~7, device+I/O ~9, full-system ~12-13 pJ/b.

---

## 2. LPDDR5 / LPDDR5X 도출 (주력 — 정밀)

### 앵커 (1차 출처, 도표 직접 판독)

**Ha, *Understanding and Improving the Energy Efficiency of DRAM*, Stanford PhD thesis 2018**
(`refs/Ha2018_DRAMEnergy_Stanford_thesis.pdf`).

- **Fig 4.8** (`refs/Ha2018_Fig4.8_LPDDR4_energy_per_bit.png`, 직접 크롭·판독): 32Gb LPDDR4 의
  `Energy/bit [pJ/bit]` = **Background + Row + Column + I/O + Refresh**.
  - SDP ~13-14 pJ/b, QDP ~10.8-13.2 pJ/b (워크로드 mix1-5).
  - 분해 (SDP mix1, 대략): Background ~1.4 · Row ~5.1 · Column ~4.1 · **I/O ~1.2** · Refresh ~2.2.
  - 캡션: "The I/O component includes energy consumed by the read driver and termination."
  - **핵심 관찰**: LPDDR 은 I/O(termination)가 작아 전체의 ~10% (~1-1.5 pJ/b)에 불과. DDR/GDDR 은 반대로 I/O가 지배적.
- **Fig 4.9**: 8Gb LPDDR4 (작은 die, 짧은 배선) ~8-9 pJ/b device-incl-I/O. 에너지/bit 는 die 용량에 비례(배선 길이).

### 스케일링 체인

| 단계 | 배수 | 근거 |
|---|---|---|
| LPDDR4 device-incl-I/O (8-16Gb) | ~10 pJ/b | Ha Fig 4.8/4.9 (8Gb ~8.5 ~ 32Gb ~13 사이) |
| -> LPDDR5 세대 이득 | x~0.75 | VDD2 1.1->1.05V, VDDQ 1.1->0.5V; JEDEC/Micron LPDDR5 ~20-30% energy/bit 개선 |
| + SoC PHY + controller | +~2-2.5 pJ/b | DRAM device I/O 는 read driver+termination 까지만; 호스트측 PHY/컨트롤러 별도 |
| **= LPDDR5-6400 full-system** | **9.0 pJ/b** | density range ~8.5(8Gb)-12.5(32Gb), 9.0 = 8-16Gb 대표 |
| -> LPDDR5X 효율 이득 | x~0.83 | Micron LPDDR5X: LPDDR5 대비 최대 ~24% 전력효율↑ (보수적 ~17% 적용) |
| **= LPDDR5X-8533 full-system** | **7.5 pJ/b** | 더 빠른 데이터레이트에 더 낮은 energy/bit |

### 교차 검증 (다른 1차 출처)

- **O'Connor et al., *Fine-Grained DRAM*, MICRO 2017** (`refs/OConnor2017...`): HBM2 = **3.97 pJ/b** full-system
  (Activation 1.21 + datapath 2.24 + interposer I/O 0.3 ...), GDDR5 = **14.0 pJ/b**, HMC = 10 pJ/b. 목표 "future 2 pJ/b".
  -> LPDDR5(9.0) 가 HBM2(3.9) 의 ~2.3x 인 것은 타당 (LPDDR 은 더 긴 PCB/패키지 배선, 더 좁은 빠른 버스).
- **Chatterjee et al., *Architecting an Energy-Efficient DRAM System for GPUs*, HPCA 2017** (`refs/Chatterjee2017...`):
  HBM column energy 1.48 pJ/b(코어) ~ 5.7 pJ/b(100% toggle), I/O 0.54 pJ/b. row/column/I/O 분해의 토글 의존성 확인.
- **Ghose et al., *Workloads and DRAM Types*, 2019** (`refs/Ghose2019...`): LPDDR3/4 가 DDR3/4 대비 DRAM 에너지를
  유의하게 절감(정성적). 절대 pJ/b 아닌 DDR3 정규화 비교라 방향성 교차검증용.
- **SemiAnalysis, *The Memory Wall* (2024)**: "DDR5 DIMM 은 read/write 에너지의 99%+ 를 host controller+interface 에서
  소비", "HBM 은 ~95% interface". full-system 경계에서 컨트롤러/인터페이스가 지배적임을 뒷받침 (특히 DDR).

> 정밀도 주의: 위 체인은 die density / SoC 컨트롤러 포함분에 따라 LPDDR5 ~8.5-12.5 pJ/b 범위. 9.0 / 7.5 는 그 중앙값.
> 랭킹 모델 목적상 ±20-30% 면 충분하나, 합성/실측 pJ 가 나오면 config `coeffs.dram` 로 덮어쓰는 게 정답.

---

## 3. DDR4 / DDR5 (보조 — secondary confidence)

LPDDR 이 검증 포커스라 DDR 은 경계 통일을 위해 같은 full-system 기준으로 맞춰만 둠. 정밀 재조사는 안 함.

- **DDR4-3200_x64: 20 pJ/b** — wide terminated x64 버스 + 컨트롤러 지배. device-incl-I/O ~15 (O'Connor GDDR5 14.0 +
  Ha 의 Micron DDR4 power-calc termination) + SoC controller.
- **DDR5-4800_x64: 14 pJ/b** — DDR4 20 x VDD 1.2->1.1V(~0.84) + on-die ECC/DFE 오버헤드.

DDR 가 주력이 되면 Micron DDR4/DDR5 System Power Calculator (IDD 기반)로 재도출 권장.

---

## 4. DRAM 선택 권고

| 용도 | 프리셋 | 비고 |
|---|---|---|
| **기본 (권장)** | `LPDDR5-6400_x16` | edge/mobile 효율 가속기 기조. 현재 `hw_config.example.json` 기본. 9.0 pJ/b |
| 더 빠른 edge | `LPDDR5X-8533_x16` | 최신 모바일/AI 노트북, +33% 대역, 7.5 pJ/b |
| 데스크탑/서버 | `DDR5-4800_x64` | 넓은 버스, secondary 신뢰도 |
| 고대역 (미래 ASIC) | (미등록) HBM2 ~3.9 pJ/b | O'Connor 1차값 있음. 필요시 `HBM2_x128` 등으로 추가 가능 |

**결론**: 스케줄러 기본 DRAM = **LPDDR5-6400_x16**, full-system **9.0 pJ/bit**. (현 FPGA xc7vx485 프로토타입엔
실제 LPDDR 컨트롤러가 없으니 이 값은 "타깃 ASIC 가정"의 모델 파라미터다.)

---

## 5. 출처 논문 (`refs/`)

| 파일 | 논문 | 이 테이블에서의 역할 |
|---|---|---|
| `Ha2018_DRAMEnergy_Stanford_thesis.pdf` | Ha, Stanford PhD 2018 | **주 앵커** (LPDDR4 full energy/bit, Fig 4.8/4.9) |
| `Ha2018_Fig4.8_LPDDR4_energy_per_bit.png` | (위 Fig 4.8 크롭) | 판독 증거 이미지 |
| `OConnor2017_FineGrainedDRAM_MICRO.pdf` | O'Connor et al., MICRO 2017 | HBM2 3.97 / GDDR5 14.0 pJ/b 교차검증 |
| `Chatterjee2017_EnergyEfficient_DRAM_GPU_HPCA.pdf` | Chatterjee et al., HPCA 2017 | row/column/I/O 분해 + 토글 의존성 |
| `Ghose2019_Workloads_DRAMTypes_arxiv1902.07609.pdf` | Ghose et al., 2019 | LPDDR vs DDR 에너지 방향성 (정성) |
| `LPSpec2025_LPDDR_PIM_LLM_arxiv2508.07227.pdf` | LP-Spec, 2025 | LPDDR PIM 최신 맥락 (참고) |

URL 은 각 파일을 받은 `curl` 출처 그대로: Ha = vlsiweb.stanford.edu, O'Connor/Chatterjee = research.nvidia.com,
Ghose/LP-Spec = arxiv.org.
