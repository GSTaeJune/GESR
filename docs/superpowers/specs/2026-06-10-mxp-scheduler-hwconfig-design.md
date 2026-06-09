# MXP_scheduler hwconfig — config 파일 기반 HW 파라미터 자동 도출 (design spec)

날짜: 2026-06-10
상태: 설계 확정 (사용자 승인 4건 반영: CACTI 실설치+호출 / chip_freq config 명시 / LPDDR5 중심 프리셋 / mac·rmw 기본값 유지)
선행: `2026-06-04-mxp-scheduler-design.md` (cost model 본체), 메모리 `project-mxp-scheduler-config-autoparam`

## 1. 목표 (한 줄)

사용자가 칩을 **물리적으로 기술**한 config 파일(SRAM 뱅크 용량·개수, DRAM 표준명, 칩 클럭)만 주면, 도구가 CACTI(SRAM)와 datasheet 프리셋 테이블(DRAM)로 모델 파라미터(`dram_bw`, `freq_ratio`, `coeffs.dram`, `coeffs.onchip`)를 **자동 도출**해서 `HW` 객체를 만든다.

## 2. 범위 / 비범위

### v1 (본 spec)

- `hw_config.json` 스키마 + 로더/검증 (`hwconfig.py`, 신규 단일 모듈)
- DRAM 프리셋 테이블 `dram_presets.json` (시드 4종) + 도출 규약
- CACTI 7 subprocess 호출 + 출력 파싱 + 결과 캐시
- 양쪽 CLI(`mxp_scheduler.py` / `mxp_scheduler_annotated.py`)에 `--config` 플래그 (플러밍만)
- CACTI 설치 (`third_party/cacti`, gitignore + 설치 가이드 문서)

### 명시적 비범위

- **Ramulator 2.0 미사용** (사용자 결정 2026-06-10). Ramulator는 타이밍 시뮬레이터라 에너지를 안 주고, 우리 stall 모델은 순수 bandwidth 모델이라 cycle-accurate 타이밍을 꽂을 자리가 없다. 보류 용도만 남김: 나중에 access-pattern별 effective-BW derate factor 캘리브레이션 or latency-aware 모델 업그레이드 시 ground truth.
- cost model 수치 로직 변경 (트윈 파일의 evaluate/dram_bits/... 은 1줄도 안 바뀜)
- mac/rmw 계수 자동 도출 (CACTI/datasheet 범위 밖 — 합성 실측값이 나오면 그때 config로 주입)
- YAML 등 JSON 외 포맷 (stdlib-only 원칙)

## 3. 사용 모습

```bash
# hw_config.json 작성 후:
python mxp_scheduler.py --config hw_config.json --M 128 --K 128 --N 128 --act 8
# 명시 CLI 플래그가 config 를 이긴다:
python mxp_scheduler.py --config hw_config.json --M 128 --K 128 --N 128 --dram-bw 32
```

우선순위: **명시 CLI 플래그 > config 도출값 > 내장 기본값**.

## 4. Config 스키마 (`hw_config.json`)

```json
{
  "sram":          {"bank_size": 1024, "banks": 32, "word_bits": 32, "tech_nm": 22},
  "dram":          "LPDDR5-6400_x16",
  "chip_freq_mhz": 250.0,
  "coeffs":        {"rmw": 5.0},
  "cacti_bin":     "third_party/cacti/cacti"
}
```

| 키 | 필수 | 의미 / 기본값 |
|---|---|---|
| `sram.bank_size` | ✔ | 뱅크당 워드 수 → `HW.bank_size` |
| `sram.banks` | ✔ | 뱅크 수 → `HW.banks` |
| `sram.word_bits` | — | 워드 폭, 기본 32 → `HW.word_bits` + CACTI 입력 |
| `sram.tech_nm` | — | CACTI 공정 노드, 기본 22 |
| `dram` | ✔ | `dram_presets.json` 의 키 (표준명) |
| `chip_freq_mhz` | ✔ | on-chip 클럭. **사용자 명시** — 칩 클럭은 SRAM 이 아니라 로직 timing closure 가 정하는 값. CACTI 의 SRAM 최대 주파수는 검증용 (§6) |
| `coeffs` | — | 자동 도출값 포함 어떤 키든 최종 오버라이드 (부분 지정 가능). 생략 시 mac/rmw 는 기존 DEFAULT (1/5) |
| `cacti_bin` | — | CACTI 실행 파일 경로. 생략 시 `CACTI_BIN` 환경변수 → PATH 의 `cacti` 순 |

검증: unknown 최상위 키 / unknown `sram.*` 키 / unknown `coeffs.*` 키 → ValueError (HW coeffs 오타 가드와 동일 철학 — 조용한 기본값 fallback 금지). 필수 키 누락 → ValueError.

## 5. DRAM 프리셋 테이블 (`dram_presets.json`)

도구와 같은 디렉토리에 두는 JSON. 사용자가 항목 추가 가능.

```json
{
  "LPDDR5-6400_x16":  {"data_rate_mts": 6400, "bus_bits": 16, "pj_per_bit": null, "source": "<채움>"},
  "LPDDR5X-8533_x16": {"data_rate_mts": 8533, "bus_bits": 16, "pj_per_bit": null, "source": "<채움>"},
  "DDR4-3200_x64":    {"data_rate_mts": 3200, "bus_bits": 64, "pj_per_bit": null, "source": "<채움>"},
  "DDR5-4800_x64":    {"data_rate_mts": 4800, "bus_bits": 64, "pj_per_bit": null, "source": "<채움>"}
}
```

- `pj_per_bit` 수치는 **구현 단계에서 문헌 조사로 확정** (datasheet IDD 환산 또는 공인 문헌값). `source` 필드에 출처 명기 필수 — null 인 항목을 resolve 가 만나면 에러 (값 없이 조용히 0 으로 돌지 않게).
- 도출 규약 (DDR 계열 공통):
  - `f_dram = data_rate_mts / 2` MHz (DDR 버스 클럭)
  - `dram_bw = 2 × bus_bits` bits/DRAM-cycle
  - `freq_ratio = chip_freq_mhz / f_dram`
  - 일관성 보장: `eff_bw = dram_bw / freq_ratio = (data_rate × bus_bits) / chip_freq` = peak bits per on-chip cycle. 예: LPDDR5-6400 x16 @ chip 250 MHz → `f_dram=3200`, `dram_bw=32`, `freq_ratio=0.078125`, `eff_bw=409.6` bits/on-chip-cycle (= 102.4 Gbit/s ÷ 250 MHz). f_dram 규약을 어떻게 잡든 eff_bw 가 물리량으로 환원되므로 모델 결과는 규약 무관.
  - peak BW 사용은 의도된 낙관 근사. effective-BW derate (row miss / RW turnaround / refresh, 통상 peak 의 70–90%) 는 비범위 — 필요해지면 Ramulator 캘리브레이션으로 (§2 비범위).
- `coeffs.dram = pj_per_bit`.

## 6. `hwconfig.py` 모듈 (트윈 아님 — 단일 파일, stdlib only)

| 함수 | 시그니처 (개념) | 역할 |
|---|---|---|
| `load_config` | `(path) -> dict` | JSON 로드 + §4 검증 |
| `dram_params` | `(name, presets_path=default) -> {dram_bw, dram_freq_mhz, pj_per_bit}` | 프리셋 lookup. 없는 이름 → 사용 가능 키 목록을 포함한 ValueError |
| `cacti_run` | `(bank_bytes, word_bits, tech_nm, cacti_bin) -> {onchip_pj_per_bit, sram_max_freq_mhz}` | CACTI .cfg 생성(단일 뱅크 SRAM, size = bank_size×word_bits/8 bytes, RW port 1) → subprocess 실행 → 출력에서 dynamic read energy per access + access time 파싱. **per-access → per-bit 환산**: `onchip_pj_per_bit = read_energy_pj / word_bits`. `sram_max_freq_mhz = 1000 / access_time_ns` |
| `resolve` | `(config, runner=cacti_run) -> dict` | 전부 결합해 `HW(...)` kwargs 반환: `bank_size, banks, word_bits, dram_bw, freq_ratio, coeffs`. coeffs 합성 순서: DEFAULT → 자동 도출(dram, onchip) → config `coeffs` 오버라이드 |

- **캐시**: `MXP_scheduler/.cacti_cache.json` (gitignore). 키 = `(bank_bytes, word_bits, tech_nm)`. 적중 시 CACTI 호출 0회. CACTI 실행은 구성당 수 초라 첫 실행만 느림.
- **runner 주입**: `resolve(config, runner=...)` 로 CACTI 호출부를 갈아끼울 수 있음 — 테스트가 fake runner 로 CACTI 없이 전 경로 검증 (§9).
- **클럭 검증**: `chip_freq_mhz > sram_max_freq_mhz` 이면 **경고 출력 후 진행** (에러 아님 — CACTI 추정과 실제 합성은 차이가 있고, 최종 판정은 Vivado timing closure 의 몫).

### 단위 일관성 (명시)

자동 모드에서 `dram`/`onchip` 계수는 실측 pJ/bit, `mac`/`rmw` 는 기존 상대 기본값(1/5) 유지. 절대 에너지 총합은 근사치가 되지만, **mac·rmw 는 매핑-상수항이라 랭킹(이 도구의 목적)에 영향이 없다** (README "mac is one scalar MAC" 절과 같은 논리). 합성/실측에서 mac·rmw 의 pJ 값이 나오면 config `coeffs` 로 주입하는 것이 업그레이드 경로.

## 7. CLI 통합 (트윈 양쪽, 플러밍만)

- `--config PATH` 플래그 추가.
- 합성 규칙 구현: HW 관련 argparse 기본값(`--bank-size 1024` 등)을 `default=None` 으로 바꾸고, `None` 이면 config 값 → config 도 없으면 기존 내장 기본값. **명시 플래그는 항상 승리.**
- `Work` 쪽(`--M/--K/--N/--act/--bits-file`)은 config 와 무관 (워크로드는 매 실행 바뀌는 값).
- cost-model 함수 무변 → `--selftest`/`--crosscheck` 영향 없음. crosscheck 의 비교 대상에 hwconfig 는 포함하지 않는다 (단일 모듈이라 트윈 drift 자체가 없음).
- config 로드/resolve 의 ValueError 는 기존과 동일하게 `p.error(...)` 로 친절하게 노출.

## 8. 에러 처리 요약

| 상황 | 동작 |
|---|---|
| config unknown/누락 키 | ValueError → `p.error` |
| unknown DRAM 이름 | 사용 가능 프리셋 목록 포함 ValueError |
| 프리셋 `pj_per_bit == null` | "출처 있는 값을 채우라"는 ValueError |
| CACTI 바이너리 못 찾음 | 설치 가이드 문서 경로를 포함한 에러 |
| CACTI 실행 실패 / 파싱 실패 | CACTI stdout 저장 경로 안내 + 에러 |
| chip_freq > SRAM max freq | 경고 출력, 계속 진행 |

## 9. 테스트 계획

CACTI 없이 도는 단위 테스트 (fake runner 주입):

1. config 로드: 정상 / unknown 키 / 필수 누락 / coeffs 부분 오버라이드
2. dram_params: 4 프리셋 도출값 (dram_bw, f_dram) golden / unknown 이름 / null pj_per_bit
3. resolve: coeffs 합성 순서 (DEFAULT → auto → config), freq_ratio 산식, eff_bw 환원 확인 (`dram_bw/freq_ratio == data_rate×bus_bits/chip_freq`)
4. 캐시: fake runner 호출 횟수 (동일 구성 2회 resolve → 1회 호출)
5. CLI 우선순위: `--config` + 명시 `--dram-bw` → 명시값 승리 (subprocess)
6. 클럭 경고: chip_freq > sram_max_freq 인 fake 결과 → 경고 문자열

CACTI 실호출 통합 테스트 1개: `pytest.mark.skipif`(CACTI 미발견) — 실제 실행 + 파싱이 양수 값을 내는지만 확인 (수치 golden 은 CACTI 버전 의존이라 두지 않음).

기존 58 케이스 / selftest / crosscheck 무회귀가 게이트.

## 10. 구현 순서 (plan 의 뼈대)

1. CACTI 7 클론+빌드 (`third_party/cacti`, gitignore) + `docs/cacti-setup.md`
2. `dram_presets.json` — pj_per_bit 문헌 조사 + source 기입
3. `hwconfig.py` (load/dram_params/cacti_run/resolve + 캐시)
4. 단위 테스트 (fake runner)
5. CLI 통합 (트윈 양쪽) + 우선순위 테스트
6. README 갱신 (config 사용법 섹션) + 통합 테스트

## 11. 파일 레이아웃 (신규)

```
MXP_scheduler/
    hwconfig.py            # 신규 — config 로더 + CACTI + DRAM 프리셋 resolve
    dram_presets.json      # 신규 — DRAM 표준 테이블 (출처 명기)
    hw_config.example.json # 신규 — 스키마 예시
    .cacti_cache.json      # 생성물 (gitignore)
third_party/cacti/         # CACTI 7 빌드 (gitignore — 클론/빌드는 setup 문서)
docs/cacti-setup.md        # 신규 — CACTI 설치 가이드
```
