#!/usr/bin/env bash
# sim/clean.sh — Remove XSim artifacts in this directory.
set -e
cd "$(dirname "$0")"
rm -rf xsim.dir webtalk*.log xvlog.log xvlog.pb xelab.log xelab.pb xsim.log xsim.pb .Xil
rm -f *.vcd *.wdb *.jou
