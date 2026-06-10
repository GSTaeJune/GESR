# MXP_scheduler/hwconfig.py
"""hwconfig — config 파일(물리 스펙) -> HW 파라미터 자동 도출 어댑터.
Spec: docs/superpowers/specs/2026-06-10-mxp-scheduler-hwconfig-design.md
트윈 아님(단일 모듈): cost-model 수치 로직은 mxp_scheduler*.py 에만 있다. stdlib only.
"""
import json
import os
import re
import subprocess
import sys
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


def _cacti_cfg(bank_bytes, word_bits, tech_nm):
    """단일 뱅크 plain-SRAM CACTI 입력.

    CACTI 의 입력 파서는 config 의 키 일부만 주면 미초기화 필드 때문에 무한루프(spin)
    또는 SIGFPE 로 죽는다 (실측: 최소 cfg → 10분+ 무출력 spin / DRAM·NUCA 키 제거 → exit 136).
    따라서 동봉된 cache.cfg 의 전체 키 세트를 그대로 따르고, 이 모델에서 의미있는 값
    (size / block size / technology / bus width / cache type=ram) 만 파라미터화한다.
    DRAM·DIMM·NUCA 키는 UCA ram 계산에 안 쓰이지만 파서가 요구하므로 보존."""
    block_bytes = word_bits // 8
    return f"""\
# parameterized: -size (bytes)
-size (bytes) {bank_bytes}
-Array Power Gating - "false"
-WL Power Gating - "false"
-CL Power Gating - "false"
-Bitline floating - "false"
-Interconnect Power Gating - "false"
-Power Gating Performance Loss 0.01
# parameterized: -block size (bytes)
-block size (bytes) {block_bytes}
-associativity 1
-read-write port 1
-exclusive read port 0
-exclusive write port 0
-single ended read ports 0
-UCA bank count 1
# parameterized: -technology (u)
-technology (u) {tech_nm / 1000}
-page size (bits) 8192
-burst length 8
-internal prefetch width 8
-Data array cell type - "itrs-hp"
-Data array peripheral type - "itrs-hp"
-Tag array cell type - "itrs-hp"
-Tag array peripheral type - "itrs-hp"
# parameterized: -output/input bus width
-output/input bus width {word_bits}
-operating temperature (K) 360
-cache type "ram"
-tag size (b) "default"
-access mode (normal, sequential, fast) - "normal"
-design objective (weight delay, dynamic power, leakage power, cycle time, area) 0:0:0:100:0
-deviate (delay, dynamic power, leakage power, cycle time, area) 20:100000:100000:100000:100000
-NUCAdesign objective (weight delay, dynamic power, leakage power, cycle time, area) 100:100:0:0:100
-NUCAdeviate (delay, dynamic power, leakage power, cycle time, area) 10:10000:10000:10000:10000
-Optimize ED or ED^2 (ED, ED^2, NONE): "ED^2"
-Cache model (NUCA, UCA)  - "UCA"
-NUCA bank count 0
-Wire signaling (fullswing, lowswing, default) - "Global_30"
-Wire inside mat - "semi-global"
-Wire outside mat - "semi-global"
-Interconnect projection - "conservative"
-Core count 8
-Cache level (L2/L3) - "L3"
-Add ECC - "true"
-Print level (DETAILED, CONCISE) - "DETAILED"
-Print input parameters - "true"
-Force cache config - "false"
-Ndwl 1
-Ndbl 1
-Nspd 0
-Ndcm 1
-Ndsam1 0
-Ndsam2 0
-dram_type "DDR3"
-io state "WRITE"
-addr_timing 1.0
-mem_density 4 Gb
-bus_freq 800 MHz
-duty_cycle 1.0
-activity_dq 1.0
-activity_ca 0.5
-num_dq 72
-num_dqs 18
-num_ca 25
-num_clk  2
-num_mem_dq 2
-mem_data_width 8
-rtt_value 10000
-ron_value 34
-tflight_value
-num_bobs 1
-capacity 80
-num_channels_per_bob 1
-first metric "Cost"
-second metric "Bandwidth"
-third metric "Energy"
-DIMM model "ALL"
-mirror_in_bob "F"
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
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}
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
                out = _HERE / "cacti_last_failure.log"
                out.write_text(f"exit code: {r.returncode}\n"
                               + r.stdout + "\n--- stderr ---\n" + r.stderr)
                raise ValueError(f"CACTI failed (exit {r.returncode}); full output saved to {out}")
    return _cached(_cache_key(bank_bytes, word_bits, tech_nm), cache_path, compute)


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
