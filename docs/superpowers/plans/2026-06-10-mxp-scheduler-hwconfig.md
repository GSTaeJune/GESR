# MXP_scheduler hwconfig Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `hw_config.json`(SRAM 스펙 + DRAM 표준명 + 칩 클럭)만 주면 CACTI와 DRAM 프리셋 테이블로 `HW` 파라미터(`dram_bw`, `freq_ratio`, `coeffs.dram/.onchip`)를 자동 도출하는 `hwconfig.py` 어댑터 + 트윈 CLI `--config` 통합.

**Architecture:** spec `docs/superpowers/specs/2026-06-10-mxp-scheduler-hwconfig-design.md` 참조. 신규 단일 모듈 `MXP_scheduler/hwconfig.py`(트윈 아님)가 config 로드 → CACTI subprocess(캐시) + `dram_presets.json` lookup → `HW` kwargs 산출. 트윈 파일에는 `--config` 플러밍만 추가 — cost-model 수치 함수 무변, crosscheck 무영향.

**Tech Stack:** Python 3 stdlib only (json, subprocess, re, pathlib). CACTI 7 (C++, `third_party/cacti`, gitignore). pytest.

**전역 규칙:**
- 모든 작업 디렉토리는 `/home/tj-home/Desktop/GESR` (repo root). pytest 는 `MXP_scheduler/` 안에서 실행.
- 매 Task 끝에 회귀 게이트: `python mxp_scheduler.py --selftest && python mxp_scheduler.py --crosscheck && python -m pytest test_mxp_scheduler.py test_hwconfig.py -q` 전부 PASS 후 커밋.
- 커밋 메시지 끝에 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: CACTI 7 설치 + setup 문서

**Files:**
- Create: `third_party/cacti/` (클론+빌드, **gitignore** — 레포에 안 들어감)
- Create: `docs/cacti-setup.md`
- Modify: `.gitignore` (repo root)

- [ ] **Step 1: CACTI 클론 + 빌드**

```bash
cd /home/tj-home/Desktop/GESR
git clone https://github.com/HewlettPackard/cacti third_party/cacti
cd third_party/cacti && make -j4
```

Expected: `third_party/cacti/cacti` 실행 파일 생성. (g++/make 필요 — 없으면 `sudo apt install build-essential` 안내 후 중단하고 사용자에게 보고. 빌드 경고는 무시, 에러만 실패.)

- [ ] **Step 2: 샘플 실행으로 동작 확인**

```bash
cd /home/tj-home/Desktop/GESR/third_party/cacti && ./cacti -infile cache.cfg | tail -30
```

Expected: `Access time (ns):` 와 `Total dynamic read energy per access (nJ):` 를 포함한 결과 블록 출력. **이 두 라인의 실제 표기 문자열을 그대로 기록해 둘 것** — Task 4 의 파서 정규식이 이 표기에 맞아야 하며, 다르면 Task 4 의 정규식을 실측 표기로 수정한다.

- [ ] **Step 3: gitignore + setup 문서**

`.gitignore` (repo root) 에 추가:

```
third_party/cacti/
MXP_scheduler/.cacti_cache.json
```

`docs/cacti-setup.md` 작성:

```markdown
# CACTI 7 설치 (MXP_scheduler hwconfig 용)

hwconfig 의 SRAM 에너지/주파수 자동 도출은 CACTI 7 바이너리가 필요하다.

​```bash
git clone https://github.com/HewlettPackard/cacti third_party/cacti
cd third_party/cacti && make -j4        # 필요: g++, make (build-essential)
./cacti -infile cache.cfg                # 동작 확인
​```

탐색 순서: config `cacti_bin` → 환경변수 `CACTI_BIN` → PATH 의 `cacti`.
결과는 `MXP_scheduler/.cacti_cache.json` 에 (bank_bytes, word_bits, tech_nm) 키로
캐시되므로 같은 SRAM 구성 재실행 시 CACTI 는 다시 돌지 않는다.

주의: CACTI 의 추정치는 공정 모델 기반 근사다. chip_freq 가 CACTI 의 SRAM 최대
주파수를 넘으면 hwconfig 가 경고를 내지만 진행은 한다 — 최종 판정은 Vivado
timing closure 의 몫.
```

- [ ] **Step 4: Commit**

```bash
git add .gitignore docs/cacti-setup.md
git commit -m "chore(scheduler): CACTI 7 setup — third_party clone guide + gitignore"
```

---

### Task 2: `dram_presets.json` + `hw_config.example.json` (+ pj/bit 출처 확정)

**Files:**
- Create: `MXP_scheduler/dram_presets.json`
- Create: `MXP_scheduler/hw_config.example.json`

- [ ] **Step 1: 잠정값으로 프리셋 작성**

`MXP_scheduler/dram_presets.json`:

```json
{
  "LPDDR5-6400_x16":  {"data_rate_mts": 6400, "bus_bits": 16, "pj_per_bit": 5.0,
                       "source": "PROVISIONAL: LPDDR4 ~8 pJ/b (Stanford Ha thesis 2018) x LPDDR5 ~0.6 power scaling"},
  "LPDDR5X-8533_x16": {"data_rate_mts": 8533, "bus_bits": 16, "pj_per_bit": 4.5,
                       "source": "PROVISIONAL: LPDDR5 5.0 pJ/b x LPDDR5X efficiency gain"},
  "DDR4-3200_x64":    {"data_rate_mts": 3200, "bus_bits": 64, "pj_per_bit": 15.0,
                       "source": "PROVISIONAL: Horowitz ISSCC14 DDR3 ~20 pJ/b scale, DDR4 improved"},
  "DDR5-4800_x64":    {"data_rate_mts": 4800, "bus_bits": 64, "pj_per_bit": 10.0,
                       "source": "PROVISIONAL: DDR4 15 pJ/b x DDR5 VDD 1.2->1.1V + arch gains"}
}
```

- [ ] **Step 2: 문헌 검증으로 값/출처 확정**

다음 두 자료에서 pJ/bit 수치를 확인하고, 확인된 값으로 JSON 의 `pj_per_bit` 를 갱신, `source` 의 `PROVISIONAL:` 접두를 제거하고 정확한 출처(저자/연도/표 번호)로 교체:

1. 저장된 PDF `/home/tj-home/.claude/projects/-home-tj-home-Desktop-GESR/a4cdb8ec-7c44-4cfd-bb68-9dea82dde661/tool-results/webfetch-1781050113582-4xlsiw.pdf` (Ha, "Understanding and Improving the Energy Efficiency of DRAM", Stanford 2018) — Read 도구의 `pages` 파라미터로 에너지 분석 장(목차에서 LPDDR4/HBM energy breakdown 장 탐색)을 읽고 LPDDR4 pJ/bit 확인.
2. WebSearch: `O'Connor "Fine-Grained DRAM" MICRO 2017 pJ/bit table` — DDR4/LPDDR4/HBM2 interface energy 표.

검증 결과가 잠정값과 2× 이상 다르면 검증값 채택. 어느 자료에서도 못 찾은 표준(LPDDR5X 등)은 가장 가까운 검증값에서의 스케일링 근거를 `source` 에 그대로 명시 (예: `"LPDDR4 8.0 pJ/b (Ha 2018 Table X) x 0.6 LPDDR5 power reduction (JEDEC)"`). **빈 출처/`PROVISIONAL` 잔류 금지.**

- [ ] **Step 3: config 예시 작성**

`MXP_scheduler/hw_config.example.json`:

```json
{
  "sram":          {"bank_size": 1024, "banks": 32, "word_bits": 32, "tech_nm": 22},
  "dram":          "LPDDR5-6400_x16",
  "chip_freq_mhz": 250.0,
  "coeffs":        {"rmw": 5.0}
}
```

- [ ] **Step 4: Commit**

```bash
git add MXP_scheduler/dram_presets.json MXP_scheduler/hw_config.example.json
git commit -m "feat(scheduler): DRAM presets table + example hw config (sourced pj/bit)"
```

---

### Task 3: `hwconfig.py` — `load_config` + `dram_params` (TDD)

**Files:**
- Create: `MXP_scheduler/hwconfig.py`
- Create: `MXP_scheduler/test_hwconfig.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`MXP_scheduler/test_hwconfig.py`:

```python
# MXP_scheduler/test_hwconfig.py
import json
import pytest
import hwconfig


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


GOOD = {"sram": {"bank_size": 1024, "banks": 32},
        "dram": "LPDDR5-6400_x16", "chip_freq_mhz": 250.0}


def test_load_config_good(tmp_path):
    cfg = hwconfig.load_config(_write(tmp_path, "c.json", GOOD))
    assert cfg["sram"]["bank_size"] == 1024
    assert cfg["sram"]["word_bits"] == 32      # default filled
    assert cfg["sram"]["tech_nm"] == 22        # default filled
    assert cfg["coeffs"] == {}                 # default filled


def test_load_config_rejects_unknown_top_key(tmp_path):
    bad = dict(GOOD); bad["dram_bw"] = 64      # belongs to CLI, not config
    with pytest.raises(ValueError, match="unknown"):
        hwconfig.load_config(_write(tmp_path, "c.json", bad))


def test_load_config_rejects_unknown_sram_key(tmp_path):
    bad = json.loads(json.dumps(GOOD)); bad["sram"]["depth"] = 7
    with pytest.raises(ValueError, match="unknown"):
        hwconfig.load_config(_write(tmp_path, "c.json", bad))


def test_load_config_rejects_missing_required(tmp_path):
    for k in ("sram", "dram", "chip_freq_mhz"):
        bad = dict(GOOD); del bad[k]
        with pytest.raises(ValueError, match="missing"):
            hwconfig.load_config(_write(tmp_path, "c.json", bad))


def test_load_config_rejects_unknown_coeff_key(tmp_path):
    bad = dict(GOOD); bad["coeffs"] = {"darm": 1.0}
    with pytest.raises(ValueError, match="unknown"):
        hwconfig.load_config(_write(tmp_path, "c.json", bad))


def test_dram_params_lpddr5():
    d = hwconfig.dram_params("LPDDR5-6400_x16")
    assert d["dram_bw"] == 32                  # 2 x bus_bits
    assert d["dram_freq_mhz"] == 3200.0        # data_rate / 2
    assert d["pj_per_bit"] > 0


def test_dram_params_unknown_lists_available():
    with pytest.raises(ValueError) as e:
        hwconfig.dram_params("HBM9")
    assert "LPDDR5-6400_x16" in str(e.value)   # error names the available presets


def test_dram_params_null_pj_rejected(tmp_path):
    p = _write(tmp_path, "presets.json",
               {"X": {"data_rate_mts": 100, "bus_bits": 8, "pj_per_bit": None, "source": "?"}})
    with pytest.raises(ValueError, match="pj_per_bit"):
        hwconfig.dram_params("X", presets_path=p)
```

- [ ] **Step 2: 실패 확인**

Run: `cd MXP_scheduler && python -m pytest test_hwconfig.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'hwconfig'`

- [ ] **Step 3: 최소 구현**

`MXP_scheduler/hwconfig.py`:

```python
# MXP_scheduler/hwconfig.py
"""hwconfig — config 파일(물리 스펙) -> HW 파라미터 자동 도출 어댑터.
Spec: docs/superpowers/specs/2026-06-10-mxp-scheduler-hwconfig-design.md
트윈 아님(단일 모듈): cost-model 수치 로직은 mxp_scheduler*.py 에만 있다. stdlib only.
"""
import json
import os
import re
import subprocess
from pathlib import Path

_HERE = Path(__file__).parent
DEFAULT_PRESETS = _HERE / "dram_presets.json"
DEFAULT_CACHE = _HERE / ".cacti_cache.json"

_TOP_KEYS = {"sram", "dram", "chip_freq_mhz", "coeffs", "cacti_bin"}
_REQUIRED = {"sram", "dram", "chip_freq_mhz"}
_SRAM_KEYS = {"bank_size", "banks", "word_bits", "tech_nm"}
_SRAM_REQUIRED = {"bank_size", "banks"}
_COEFF_KEYS = {"dram", "onchip", "mac", "rmw"}


def load_config(path):
    """hw_config.json 로드 + 검증 + 기본값 채움. 모든 위반은 ValueError (CLI 가 p.error 로 노출)."""
    with open(path) as f:
        cfg = json.load(f)
    unknown = set(cfg) - _TOP_KEYS
    if unknown:
        raise ValueError(f"config has unknown key(s) {sorted(unknown)}; valid: {sorted(_TOP_KEYS)}")
    missing = _REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"config missing required key(s) {sorted(missing)}")
    sram = cfg["sram"]
    unknown = set(sram) - _SRAM_KEYS
    if unknown:
        raise ValueError(f"config sram has unknown key(s) {sorted(unknown)}; valid: {sorted(_SRAM_KEYS)}")
    missing = _SRAM_REQUIRED - set(sram)
    if missing:
        raise ValueError(f"config sram missing required key(s) {sorted(missing)}")
    sram.setdefault("word_bits", 32)
    sram.setdefault("tech_nm", 22)
    cfg.setdefault("coeffs", {})
    unknown = set(cfg["coeffs"]) - _COEFF_KEYS
    if unknown:
        raise ValueError(f"config coeffs has unknown key(s) {sorted(unknown)}; valid: {sorted(_COEFF_KEYS)}")
    return cfg


def dram_params(name, presets_path=DEFAULT_PRESETS):
    """프리셋 lookup -> {dram_bw, dram_freq_mhz, pj_per_bit}.
    규약(spec §5): f_dram = data_rate/2 (DDR 버스 클럭), dram_bw = 2*bus_bits."""
    with open(presets_path) as f:
        presets = json.load(f)
    if name not in presets:
        raise ValueError(f"unknown DRAM {name!r}; available: {sorted(presets)}")
    p = presets[name]
    if not p.get("pj_per_bit"):
        raise ValueError(f"preset {name!r} has no pj_per_bit — fill it with a sourced value "
                         f"(source field) in {presets_path}")
    return {"dram_bw": 2 * p["bus_bits"],
            "dram_freq_mhz": p["data_rate_mts"] / 2.0,
            "pj_per_bit": float(p["pj_per_bit"])}
```

- [ ] **Step 4: 통과 확인**

Run: `cd MXP_scheduler && python -m pytest test_hwconfig.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/hwconfig.py MXP_scheduler/test_hwconfig.py
git commit -m "feat(scheduler): hwconfig load_config + dram_params (TDD)"
```

---

### Task 4: `hwconfig.py` — CACTI cfg 생성 / 출력 파싱 / 실행 + 캐시 (TDD)

**Files:**
- Modify: `MXP_scheduler/hwconfig.py` (함수 추가)
- Modify: `MXP_scheduler/test_hwconfig.py` (테스트 추가)

- [ ] **Step 1: 실패하는 테스트 작성** — `test_hwconfig.py` 에 추가:

```python
# --- Task 4: CACTI cfg/parse/cache ---

CACTI_SAMPLE_OUT = """
... (preamble) ...
    Access time (ns): 0.853214
    Cycle time (ns):  0.41
    Total dynamic read energy per access (nJ): 0.00482
    Total dynamic write energy per access (nJ): 0.00513
... (rest) ...
"""


def test_cacti_cfg_contains_size_and_tech():
    cfg = hwconfig._cacti_cfg(bank_bytes=4096, word_bits=32, tech_nm=22)
    assert '-size (bytes) 4096' in cfg
    assert '-output/input bus width 32' in cfg
    assert '-technology (u) 0.022' in cfg
    assert '"ram"' in cfg                      # plain SRAM, not cache


def test_parse_cacti_per_bit_conversion():
    r = hwconfig._parse_cacti(CACTI_SAMPLE_OUT, word_bits=32)
    # 0.00482 nJ/access = 4.82 pJ/access -> /32 bits = 0.150625 pJ/bit
    assert r["onchip_pj_per_bit"] == pytest.approx(4.82 / 32)
    assert r["sram_max_freq_mhz"] == pytest.approx(1000.0 / 0.853214)


def test_parse_cacti_failure_raises():
    with pytest.raises(ValueError, match="parse"):
        hwconfig._parse_cacti("no relevant lines here", word_bits=32)


def test_cache_roundtrip(tmp_path):
    cache = str(tmp_path / "cache.json")
    calls = []
    def fake_compute():
        calls.append(1)
        return {"onchip_pj_per_bit": 0.15, "sram_max_freq_mhz": 1172.0}
    k = hwconfig._cache_key(4096, 32, 22)
    r1 = hwconfig._cached(k, cache, fake_compute)
    r2 = hwconfig._cached(k, cache, fake_compute)
    assert r1 == r2 and len(calls) == 1        # 2nd hit served from cache file
    r3 = hwconfig._cached(hwconfig._cache_key(8192, 32, 22), cache, fake_compute)
    assert len(calls) == 2                     # different key -> recompute


def test_find_cacti_bin_missing_gives_guide(monkeypatch, tmp_path):
    monkeypatch.delenv("CACTI_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))  # empty PATH dir
    with pytest.raises(ValueError, match="cacti-setup"):
        hwconfig._find_cacti_bin(None)
```

- [ ] **Step 2: 실패 확인**

Run: `cd MXP_scheduler && python -m pytest test_hwconfig.py -q`
Expected: 신규 5개 FAIL (`AttributeError: ... _cacti_cfg`), 기존 8개 PASS

- [ ] **Step 3: 구현** — `hwconfig.py` 에 추가:

```python
def _cacti_cfg(bank_bytes, word_bits, tech_nm):
    """단일 뱅크 plain-SRAM CACTI 입력. CACTI 의 cache.cfg 포맷 중 필요한 키만 명시."""
    return f"""
-size (bytes) {bank_bytes}
-block size (bytes) {word_bits // 8}
-associativity 1
-read-write port 1
-exclusive read port 0
-exclusive write port 0
-UCA bank count 1
-technology (u) {tech_nm / 1000}
-output/input bus width {word_bits}
-cache type "ram"
-tag size (b) "default"
-access mode (normal, sequential, fast) "normal"
-operating temperature (K) 350
-Cache model (NUCA, UCA)  "UCA"
-design objective (weight delay, dynamic power, leakage power, cycle time, area) 0:0:0:100:0
-deviate (delay, dynamic power, leakage power, cycle time, area) 20:100000:100000:100000:100000
-Optimize ED or ED^2 (ED, ED^2, NONE): "ED^2"
-Print level (DETAILED, CONCISE) -- "DETAILED"
-internal prefetch width 8
-Data array cell type - "itrs-hp"
-Data array peripheral type - "itrs-hp"
"""


def _parse_cacti(text, word_bits):
    """CACTI stdout 에서 access time / read energy 추출 -> per-bit 환산.
    표기 차이(버전)에 견디도록 라인 단위 regex. 실패 시 ValueError."""
    at = re.search(r"Access time \(ns\):\s*([\d.eE+-]+)", text)
    re_ = re.search(r"Total dynamic read energy per access \(nJ\):\s*([\d.eE+-]+)", text)
    if not (at and re_):
        raise ValueError("could not parse CACTI output (Access time / read energy lines missing)")
    read_pj = float(re_.group(1)) * 1000.0          # nJ -> pJ per access
    return {"onchip_pj_per_bit": read_pj / word_bits,
            "sram_max_freq_mhz": 1000.0 / float(at.group(1))}


def _cache_key(bank_bytes, word_bits, tech_nm):
    return f"{bank_bytes}b_{word_bits}w_{tech_nm}nm"


def _cached(key, cache_path, compute):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    if key not in cache:
        cache[key] = compute()
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=1)
    return cache[key]


def _find_cacti_bin(cfg_bin):
    """config cacti_bin -> $CACTI_BIN -> PATH 순. 못 찾으면 설치 가이드 경로 포함 에러."""
    import shutil
    for cand in (cfg_bin, os.environ.get("CACTI_BIN")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    found = shutil.which("cacti")
    if found:
        return found
    raise ValueError("CACTI binary not found (config cacti_bin / $CACTI_BIN / PATH). "
                     "Install guide: docs/cacti-setup.md")


def cacti_run(bank_bytes, word_bits, tech_nm, cacti_bin=None, cache_path=DEFAULT_CACHE):
    """CACTI 실행(캐시 적중 시 생략) -> {onchip_pj_per_bit, sram_max_freq_mhz}."""
    def compute():
        binp = _find_cacti_bin(cacti_bin)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            cfgp = Path(td) / "sram.cfg"
            cfgp.write_text(_cacti_cfg(bank_bytes, word_bits, tech_nm))
            r = subprocess.run([os.path.abspath(binp), "-infile", str(cfgp)],
                               capture_output=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(binp)))
            try:
                return _parse_cacti(r.stdout, word_bits)
            except ValueError:
                dump = Path(td).with_suffix("")  # td 는 곧 사라지므로 옆에 저장
                out = _HERE / "cacti_last_failure.log"
                out.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
                raise ValueError(f"CACTI output parse failed; full output saved to {out}")
    return _cached(_cache_key(bank_bytes, word_bits, tech_nm), cache_path, compute)
```

(주의: CACTI 는 자기 디렉토리의 부속 파일을 상대경로로 읽으므로 `cwd` 를 바이너리 위치로 둔다. Task 1 Step 2 에서 기록한 실측 출력 표기가 위 regex 와 다르면 **regex 를 실측 표기에 맞춰 수정** — `CACTI_SAMPLE_OUT` 테스트 픽스처도 같은 표기로 갱신.)

- [ ] **Step 4: 통과 확인**

Run: `cd MXP_scheduler && python -m pytest test_hwconfig.py -q`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/hwconfig.py MXP_scheduler/test_hwconfig.py
git commit -m "feat(scheduler): hwconfig CACTI cfg/parse/run + result cache (TDD)"
```

---

### Task 5: `resolve` — 전체 결합 + coeffs 합성 + 클럭 경고 (TDD)

**Files:**
- Modify: `MXP_scheduler/hwconfig.py`
- Modify: `MXP_scheduler/test_hwconfig.py`

- [ ] **Step 1: 실패하는 테스트 작성** — `test_hwconfig.py` 에 추가:

```python
# --- Task 5: resolve ---

def fake_runner(bank_bytes, word_bits, tech_nm, cacti_bin=None, cache_path=None):
    return {"onchip_pj_per_bit": 0.15, "sram_max_freq_mhz": 1000.0}


def _resolve(tmp_path, cfg_overrides=None, runner=fake_runner):
    cfg = json.loads(json.dumps(GOOD))
    if cfg_overrides:
        cfg.update(cfg_overrides)
    path = _write(tmp_path, "c.json", cfg)
    return hwconfig.resolve(hwconfig.load_config(path), runner=runner)


def test_resolve_hw_kwargs(tmp_path):
    kw = _resolve(tmp_path)
    assert kw["bank_size"] == 1024 and kw["banks"] == 32 and kw["word_bits"] == 32
    assert kw["dram_bw"] == 32                                   # LPDDR5-6400_x16
    assert kw["freq_ratio"] == pytest.approx(250.0 / 3200.0)
    # eff_bw 환원: dram_bw/freq_ratio == data_rate*bus_bits/chip_freq = 6400*16/250
    assert kw["dram_bw"] / kw["freq_ratio"] == pytest.approx(6400 * 16 / 250.0)


def test_resolve_coeff_layering(tmp_path):
    kw = _resolve(tmp_path, {"coeffs": {"rmw": 7.0}})
    c = kw["coeffs"]
    assert c["dram"] > 0 and c["dram"] != 200.0   # auto from preset, not DEFAULT
    assert c["onchip"] == pytest.approx(0.15)      # auto from (fake) CACTI
    assert c["mac"] == 1.0                         # DEFAULT preserved
    assert c["rmw"] == 7.0                         # config override wins


def test_resolve_warns_when_chip_faster_than_sram(tmp_path, capsys):
    slow_sram = lambda *a, **k: {"onchip_pj_per_bit": 0.15, "sram_max_freq_mhz": 200.0}
    _resolve(tmp_path, runner=slow_sram)           # chip 250 > sram 200
    assert "warning" in capsys.readouterr().err.lower()


def test_resolve_valid_for_HW(tmp_path):
    import mxp_scheduler as s
    hw = s.HW(**_resolve(tmp_path))                # kwargs must construct a valid HW
    assert hw.cap_bits == 1024 * 32 * 32
```

- [ ] **Step 2: 실패 확인**

Run: `cd MXP_scheduler && python -m pytest test_hwconfig.py -q`
Expected: 신규 4개 FAIL (`AttributeError: ... resolve`), 기존 13개 PASS

- [ ] **Step 3: 구현** — `hwconfig.py` 에 추가 (DEFAULT_COEFFS 는 트윈에 의존하지 않게 자체 보유 — 값은 mxp_scheduler.DEFAULT_COEFFS 와 동일해야 하며 Task 6 테스트가 일치를 잠근다):

```python
import sys

DEFAULT_COEFFS = {"dram": 200.0, "onchip": 6.0, "mac": 1.0, "rmw": 5.0}


def resolve(cfg, runner=cacti_run, presets_path=DEFAULT_PRESETS, cache_path=DEFAULT_CACHE):
    """검증된 config dict -> HW(...) kwargs.
    coeffs 합성: DEFAULT -> 자동 도출(dram=프리셋 pJ/bit, onchip=CACTI pJ/bit) -> config coeffs."""
    sram, chip = cfg["sram"], cfg["chip_freq_mhz"]
    d = dram_params(cfg["dram"], presets_path)
    bank_bytes = sram["bank_size"] * sram["word_bits"] // 8
    c = runner(bank_bytes, sram["word_bits"], sram["tech_nm"],
               cacti_bin=cfg.get("cacti_bin"), cache_path=cache_path)
    if chip > c["sram_max_freq_mhz"]:
        print(f"warning: chip_freq {chip} MHz > CACTI SRAM max "
              f"{c['sram_max_freq_mhz']:.0f} MHz — timing closure will decide", file=sys.stderr)
    coeffs = dict(DEFAULT_COEFFS)
    coeffs["dram"] = d["pj_per_bit"]
    coeffs["onchip"] = c["onchip_pj_per_bit"]
    coeffs.update(cfg["coeffs"])
    return {"bank_size": sram["bank_size"], "banks": sram["banks"],
            "word_bits": sram["word_bits"], "dram_bw": float(d["dram_bw"]),
            "freq_ratio": chip / d["dram_freq_mhz"], "coeffs": coeffs}
```

- [ ] **Step 4: 통과 확인**

Run: `cd MXP_scheduler && python -m pytest test_hwconfig.py -q`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add MXP_scheduler/hwconfig.py MXP_scheduler/test_hwconfig.py
git commit -m "feat(scheduler): hwconfig resolve — HW kwargs synthesis + coeff layering (TDD)"
```

---

### Task 6: CLI `--config` 통합 (트윈 양쪽) + 우선순위 테스트

**Files:**
- Modify: `MXP_scheduler/mxp_scheduler.py` (`main()` 만)
- Modify: `MXP_scheduler/mxp_scheduler_annotated.py` (`main()` 만 — 한국어 주석으로 미러)
- Modify: `MXP_scheduler/test_hwconfig.py`

- [ ] **Step 1: 실패하는 테스트 작성** — `test_hwconfig.py` 에 추가:

```python
# --- Task 6: CLI integration ---
import pathlib
import subprocess as sp
import sys as _sys

HERE = pathlib.Path(__file__).parent


def _run_cli(args, script="mxp_scheduler.py"):
    return sp.run([_sys.executable, script] + args, cwd=HERE, capture_output=True, text=True)


def _cfg_file(tmp_path):
    # 실 CACTI 없이 CLI 를 테스트하기 위해 캐시를 미리 심는다 (cache key 는 4096b_32w_22nm)
    cache = {"4096b_32w_22nm": {"onchip_pj_per_bit": 0.15, "sram_max_freq_mhz": 1000.0}}
    (HERE / ".cacti_cache.json").write_text(json.dumps(cache))
    return _write(tmp_path, "hw.json", GOOD)


def test_cli_config_drives_hw(tmp_path):
    r = _run_cli(["--config", _cfg_file(tmp_path), "--M", "64", "--K", "64", "--N", "64"])
    assert r.returncode == 0, r.stderr
    head = r.stdout.splitlines()[0]
    assert "freq_ratio=0.078125" in head        # 250/3200 — config 가 실제로 적용됨


def test_cli_explicit_flag_beats_config(tmp_path):
    r = _run_cli(["--config", _cfg_file(tmp_path), "--M", "64", "--K", "64", "--N", "64",
                  "--dram-bw", "8"])
    assert r.returncode == 0, r.stderr
    assert "dram_bw=8.0" in r.stdout.splitlines()[0]   # 명시 플래그 승리


def test_cli_no_config_unchanged(tmp_path):
    r = _run_cli(["--M", "64", "--K", "64", "--N", "64"])
    assert r.returncode == 0, r.stderr
    head = r.stdout.splitlines()[0]
    assert "dram_bw=64.0" in head and "freq_ratio=1.0" in head   # 기존 기본값 그대로


def test_cli_annotated_config_same_result(tmp_path):
    cfgp = _cfg_file(tmp_path)
    r1 = _run_cli(["--config", cfgp, "--M", "64", "--K", "64", "--N", "64"])
    r2 = _run_cli(["--config", cfgp, "--M", "64", "--K", "64", "--N", "64"],
                  script="mxp_scheduler_annotated.py")
    assert r1.stdout == r2.stdout               # 트윈 동일 출력


def test_default_coeffs_match_twin():
    import mxp_scheduler as s
    assert hwconfig.DEFAULT_COEFFS == s.DEFAULT_COEFFS
```

- [ ] **Step 2: 실패 확인**

Run: `cd MXP_scheduler && python -m pytest test_hwconfig.py -q`
Expected: 신규 5개 중 `test_cli_config_drives_hw`/`_beats_config`/`_annotated` FAIL (`--config` 미존재로 argparse 에러), `_no_config_unchanged`/`_match_twin` PASS

- [ ] **Step 3: 표준판 `main()` 수정** — `mxp_scheduler.py` 의 main 에서 HW 플래그 기본값을 None 으로 바꾸고 합성:

```python
    p.add_argument("--config", help="hw_config.json — physical HW description (see hwconfig.py)")
    p.add_argument("--bank-size", type=int, default=None)
    p.add_argument("--banks", type=int, default=None)
    p.add_argument("--dram-bw", type=float, default=None, help="bits per DRAM cycle")
    p.add_argument("--freq-ratio", type=float, default=None,
                   help="on-chip cycles per DRAM cycle (f_chip/f_dram); 1.0 = same clock")
```

기존 Work/HW 생성부를 다음으로 교체 (try 블록 포함):

```python
    coeffs_file = {}
    if a.coeffs:
        import json
        with open(a.coeffs) as f:
            coeffs_file = json.load(f)
    try:
        cfg_kw = None
        if a.config:
            import hwconfig
            cfg_kw = hwconfig.resolve(hwconfig.load_config(a.config))
        # 우선순위: 명시 CLI 플래그 > config 도출값 > 내장 기본값
        def pick(cli_val, key, builtin):
            if cli_val is not None:
                return cli_val
            return cfg_kw[key] if cfg_kw is not None else builtin
        coeffs = dict(cfg_kw["coeffs"]) if cfg_kw is not None else dict(DEFAULT_COEFFS)
        coeffs.update(coeffs_file)                      # --coeffs 는 항상 최종 승리
        wbits = _load_wbits(a.bits_file, MT, KT) if a.bits_file else [[a.act] * KT for _ in range(MT)]
        w = Work(M=a.M, K=a.K, N=a.N, wbits=wbits, act_bits=a.act)
        hw = HW(bank_size=pick(a.bank_size, "bank_size", 1024),
                banks=pick(a.banks, "banks", 32),
                dram_bw=pick(a.dram_bw, "dram_bw", 64.0),
                word_bits=cfg_kw["word_bits"] if cfg_kw is not None else 32,
                freq_ratio=pick(a.freq_ratio, "freq_ratio", 1.0),
                coeffs=coeffs)
    except ValueError as e:
        p.error(str(e))
```

(`MT, KT = a.M // TILE, a.K // TILE` 줄은 그 위에 유지. 기존 `coeffs = dict(DEFAULT_COEFFS)` / `coeffs.update(...)` 블록은 위 코드로 흡수되므로 삭제.)

- [ ] **Step 4: annotated `main()` 에 동일 수정** — 같은 코드를 한국어 주석으로 미러 (annotated 는 `--explain` 플래그 분기 유지, `--selftest/--crosscheck` 없음). 주석 예: `# 우선순위: 명시 CLI 플래그 > config 도출값 > 내장 기본값 (spec §7)`.

- [ ] **Step 5: 전체 통과 + 회귀 확인**

Run: `cd MXP_scheduler && python -m pytest test_hwconfig.py test_mxp_scheduler.py -q && python mxp_scheduler.py --selftest && python mxp_scheduler.py --crosscheck`
Expected: 22 + 58 = 80 passed, selftest OK, crosscheck OK

- [ ] **Step 6: Commit**

```bash
git add MXP_scheduler/mxp_scheduler.py MXP_scheduler/mxp_scheduler_annotated.py MXP_scheduler/test_hwconfig.py
git commit -m "feat(scheduler): --config flag — config-driven HW params in both twin CLIs"
```

---

### Task 7: CACTI 실호출 통합 테스트 + README + 마무리

**Files:**
- Modify: `MXP_scheduler/test_hwconfig.py`
- Modify: `MXP_scheduler/README.md`

- [ ] **Step 1: 실 CACTI 통합 테스트 추가** — `test_hwconfig.py` 에 추가:

```python
# --- Task 7: real-CACTI integration (skipped when CACTI absent) ---

def _cacti_available():
    try:
        hwconfig._find_cacti_bin(str(HERE.parent / "third_party" / "cacti" / "cacti"))
        return True
    except ValueError:
        return False


@pytest.mark.skipif(not _cacti_available(), reason="CACTI binary not installed")
def test_real_cacti_run(tmp_path):
    r = hwconfig.cacti_run(4096, 32, 22,
                           cacti_bin=str(HERE.parent / "third_party" / "cacti" / "cacti"),
                           cache_path=str(tmp_path / "c.json"))
    assert r["onchip_pj_per_bit"] > 0          # 수치 golden 은 CACTI 버전 의존이라 두지 않음
    assert r["sram_max_freq_mhz"] > 0
```

Run: `cd MXP_scheduler && python -m pytest test_hwconfig.py -q`
Expected: 23 passed (CACTI 설치돼 있으면 real 테스트 포함; 파싱 실패 시 Task 4 의 regex 를 `cacti_last_failure.log` 의 실측 표기로 수정)

- [ ] **Step 2: README 에 config 섹션 추가** — `MXP_scheduler/README.md` 의 CLI 섹션 뒤에:

```markdown
## `--config`: 물리 스펙으로 HW 파라미터 자동 도출

`hw_config.json` 에 칩을 물리적으로 기술하면 CACTI(SRAM)와 `dram_presets.json`(DRAM
datasheet 값)으로 `dram_bw / freq_ratio / coeffs.dram / coeffs.onchip` 을 자동 도출한다.
명시 CLI 플래그는 항상 config 를 이긴다. CACTI 설치: `docs/cacti-setup.md`.

​```bash
cp hw_config.example.json hw_config.json   # 편집: SRAM 스펙 / DRAM 표준명 / 칩 클럭
python mxp_scheduler.py --config hw_config.json --M 128 --K 128 --N 128 --act 8
​```

mac/rmw 계수는 자동 도출 범위 밖(로직 에너지)이라 기본값이 유지된다 — 매핑-상수항이라
랭킹에는 영향 없음. 실측값이 생기면 config `coeffs` 로 주입.
```

테스트 수 표기도 갱신: `# unit suite (58 cases)` → `(58 cases)` 유지 + 아래 줄 추가 `python -m pytest test_hwconfig.py -q   # hwconfig suite (23 cases)`.

- [ ] **Step 3: 전체 회귀 + 커밋**

Run: `cd MXP_scheduler && python -m pytest -q && python mxp_scheduler.py --selftest && python mxp_scheduler.py --crosscheck`
Expected: 81 passed, selftest OK, crosscheck OK

```bash
git add MXP_scheduler/test_hwconfig.py MXP_scheduler/README.md
git commit -m "feat(scheduler): hwconfig real-CACTI integration test + README config section"
```

---

## Self-Review 결과 (작성 직후 점검)

- **Spec coverage**: §4 스키마→Task 3, §5 프리셋/규약→Task 2·3, §6 모듈/캐시/경고→Task 4·5, §7 CLI→Task 6, §8 에러→Task 3·4·5 테스트, §9 테스트 계획→Task 3~7 (fake runner / 캐시 횟수 / 우선순위 / 경고 / skipif 실호출 전부 매핑), §10 순서 일치, §11 레이아웃→Task 1·2. 누락 없음.
- **Placeholder**: Task 2 의 `PROVISIONAL` 은 Step 2 가 제거를 강제 (잔류 금지 명시) — 의도된 2단계.
- **Type consistency**: `_cache_key/_cached/cacti_run/resolve` 시그니처가 Task 4 정의 = Task 5·6·7 사용처 일치. `fake_runner` 시그니처 = `runner(bank_bytes, word_bits, tech_nm, cacti_bin=, cache_path=)` 일치. CACTI cfg 키 표기는 CACTI 7 cache.cfg 기준 — Task 1 Step 2 실측과 다르면 Task 4 에서 수정하라는 지침 포함.
