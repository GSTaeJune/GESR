#!/usr/bin/env python3
"""Timeloop v4 smoke run: GEMM accelerator (W/A/O dedicated buffers + 32x32 SA).

Run inside WSL, conda env `timeloop` (see run_smoke.sh).
Outputs land in ./output/ next to this file.
"""
import os
import sys

import timeloopfe.v4 as tl  # this install ships the standalone `timeloopfe` package
                            # (newer installs name it `pytimeloop.timeloopfe.v4`)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

spec = tl.Specification.from_yaml_files(
    os.path.join(HERE, "arch.yaml"),
    os.path.join(HERE, "components", "smartbuffer_SRAM.yaml"),
    os.path.join(HERE, "components", "intmac.yaml"),
    os.path.join(HERE, "problem.yaml"),
    os.path.join(HERE, "mapper.yaml"),
)

tl.call_mapper(spec, output_dir=OUT, dump_intermediate_to=OUT)

stats = os.path.join(OUT, "timeloop-mapper.stats.txt")
if not os.path.exists(stats):
    print("FAIL: no stats produced", file=sys.stderr)
    sys.exit(1)
print("OK:", stats)
