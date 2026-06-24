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


# --- Task 6: CLI integration ---
import pathlib
import subprocess as sp
import sys as _sys

HERE = pathlib.Path(__file__).parent


def _run_cli(args, script="mxp_scheduler.py"):
    return sp.run([_sys.executable, script] + args, cwd=HERE, capture_output=True, text=True)


def _cfg_file(tmp_path):
    return _write(tmp_path, "hw.json", GOOD)


@pytest.fixture
def planted_cache():
    """CLI 테스트용: 실제 캐시 경로에 스텁을 심되, 끝나면 원상 복구 (테스트가 실 캐시를 오염시키지 않게)."""
    p = HERE / ".cacti_cache.json"
    orig = p.read_text() if p.exists() else None
    p.write_text(json.dumps(
        {"4096b_32w_22nm": {"onchip_pj_per_bit": 0.15, "sram_max_freq_mhz": 1000.0}}))
    yield
    if orig is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text(orig)


def test_cli_config_drives_hw(tmp_path, planted_cache):
    r = _run_cli(["--config", _cfg_file(tmp_path), "--M", "64", "--K", "64", "--N", "64"])
    assert r.returncode == 0, r.stderr
    head = r.stdout.splitlines()[0]
    assert "freq_ratio=0.078125" in head        # 250/3200 — config 가 실제로 적용됨


def test_cli_explicit_flag_beats_config(tmp_path, planted_cache):
    r = _run_cli(["--config", _cfg_file(tmp_path), "--M", "64", "--K", "64", "--N", "64",
                  "--dram-bw", "8"])
    assert r.returncode == 0, r.stderr
    assert "dram_bw=8.0" in r.stdout.splitlines()[0]   # 명시 플래그 승리


def test_cli_no_config_unchanged(tmp_path):
    r = _run_cli(["--M", "64", "--K", "64", "--N", "64"])
    assert r.returncode == 0, r.stderr
    head = r.stdout.splitlines()[0]
    assert "dram_bw=64.0" in head and "freq_ratio=1.0" in head   # 기존 기본값 그대로


def test_cli_annotated_config_same_result(tmp_path, planted_cache):
    cfgp = _cfg_file(tmp_path)
    r1 = _run_cli(["--config", cfgp, "--M", "64", "--K", "64", "--N", "64"])
    r2 = _run_cli(["--config", cfgp, "--M", "64", "--K", "64", "--N", "64"],
                  script="mxp_scheduler_annotated.py")
    assert r1.stdout == r2.stdout               # 트윈 동일 출력
    # flag-wins 경로도 동일 — 명시 플래그가 config 를 덮어쓰는 동작이 양쪽에서 일치
    f1 = _run_cli(["--config", cfgp, "--M", "64", "--K", "64", "--N", "64", "--dram-bw", "8"])
    f2 = _run_cli(["--config", cfgp, "--M", "64", "--K", "64", "--N", "64", "--dram-bw", "8"],
                  script="mxp_scheduler_annotated.py")
    assert f1.stdout == f2.stdout               # 트윈 동일 출력 (flag-wins)


def test_default_coeffs_match_twin():
    import mxp_scheduler as s
    assert hwconfig.DEFAULT_COEFFS == s.DEFAULT_COEFFS


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


def test_resolve_passes_cycle_params(tmp_path, monkeypatch):
    import hwconfig as hc
    cfg = {"sram": {"bank_size": 1024, "banks": 32}, "dram": "LPDDR5-6400_x16",
           "chip_freq_mhz": 250.0, "cycles": {"cycles_per_bit": 2.0, "sa_fill_cycles": 5}}
    p = tmp_path / "hw.json"
    p.write_text(__import__("json").dumps(cfg))
    loaded = hc.load_config(str(p))
    kw = hc.resolve(loaded, runner=lambda *a, **k: {"onchip_pj_per_bit": 0.85, "sram_max_freq_mhz": 900.0})
    assert kw["cycles_per_bit"] == 2.0
    assert kw["sa_fill_cycles"] == 5
    assert kw["sa_drain_cycles"] == 0   # default


def test_load_config_rejects_unknown_cycle_key(tmp_path):
    import hwconfig as hc, pytest
    cfg = {"sram": {"bank_size": 1, "banks": 1}, "dram": "X", "chip_freq_mhz": 1,
           "cycles": {"cyclez_per_bit": 2.0}}
    p = tmp_path / "hw.json"
    p.write_text(__import__("json").dumps(cfg))
    with pytest.raises(ValueError):
        hc.load_config(str(p))


def test_dram_presets_schema_and_provenance():
    import json, hwconfig as hc
    presets = json.load(open(hc.DEFAULT_PRESETS))
    assert presets, "dram_presets.json must be non-empty"
    for name, p in presets.items():
        assert set(p) >= {"data_rate_mts", "bus_bits", "pj_per_bit", "source"}, \
            f"{name} missing required field(s)"
        assert p["data_rate_mts"] > 0 and p["bus_bits"] > 0 and p["pj_per_bit"] > 0, \
            f"{name} has non-positive numeric field"
        assert isinstance(p["source"], str) and len(p["source"].strip()) >= 20, \
            f"{name} source provenance too short/missing"

def test_dram_params_derivation_matches_convention():
    import hwconfig as hc
    d = hc.dram_params("LPDDR5-6400_x16")
    # spec convention: f_dram = data_rate/2 (DDR bus clock); dram_bw = 2*bus_bits
    assert d["dram_bw"] == 2 * 16
    assert d["dram_freq_mhz"] == 6400 / 2.0
    assert d["pj_per_bit"] == 9.0
