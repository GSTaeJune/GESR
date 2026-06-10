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
