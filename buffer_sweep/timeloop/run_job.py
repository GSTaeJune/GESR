# buffer_sweep/timeloop/run_job.py
"""Run timeloop-mapper for ONE sweep job dir (arch.yaml + problem.yaml staged
next to this script by run_jobs.sh). Writes ./output/. A failed search leaves
no map.yaml; run_jobs.sh turns that into an INVALID marker (NOTES.md pitfall 3:
'Could not find cycles in stats' == no valid mapping, not a crash)."""
import os
import timeloopfe.v4 as tl

os.makedirs("output", exist_ok=True)
spec = tl.Specification.from_yaml_files(
    "arch.yaml",
    "components/smartbuffer_SRAM.yaml",
    "components/intmac.yaml",
    "problem.yaml",
    "mapper_sweep.yaml",
)
tl.call_mapper(spec, output_dir="output", dump_intermediate_to="output")
