# CACTI 7 설치 (MXP_scheduler hwconfig 용)

hwconfig 의 SRAM 에너지/주파수 자동 도출은 CACTI 7 바이너리가 필요하다.

```bash
git clone https://github.com/HewlettPackard/cacti third_party/cacti
cd third_party/cacti && make -j4        # 필요: g++, make (build-essential)
./cacti -infile cache.cfg                # 동작 확인
```

탐색 순서: config `cacti_bin` → 환경변수 `CACTI_BIN` → PATH 의 `cacti`.
결과는 `MXP_scheduler/.cacti_cache.json` 에 (bank_bytes, word_bits, tech_nm) 키로
캐시되므로 같은 SRAM 구성 재실행 시 CACTI 는 다시 돌지 않는다.

주의: CACTI 의 추정치는 공정 모델 기반 근사다. chip_freq 가 CACTI 의 SRAM 최대
주파수를 넘으면 hwconfig 가 경고를 내지만 진행은 한다 — 최종 판정은 Vivado
timing closure 의 몫.

## 빌드 노트 (검증: g++ 13.3, Ubuntu 24.04)

makefile 수정 없이 빌드 성공. 빌드 중 경고는 정상이며 무시해도 된다:
- `g++: warning: switch '-gstabs+' is no longer supported` — CACTI 의 디버그 플래그가
  구식이라 최신 g++ 가 무시. 빌드에 영향 없음.
- `-Wparentheses` 등 코드 경고 다수 — CACTI 7 은 오래된 C++ 코드라 정상.

빌드 산출물: `third_party/cacti/cacti` 실행 파일.

## 파서가 의존하는 출력 라인 (Task 4 참조)

`./cacti -infile cache.cfg` 출력에서 hwconfig 가 읽는 라인 원문 (앞 공백 포함, 4-space 들여쓰기):

```
    Access time (ns): 1.47098
    Total dynamic read energy per access (nJ): 0.303592
```

(쓰기 에너지가 필요하면 `    Total dynamic write energy per access (nJ): 0.615022` 도 같은 포맷.)
