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
