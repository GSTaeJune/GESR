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
