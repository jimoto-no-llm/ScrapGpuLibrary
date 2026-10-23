#!/bin/bash
cd "$(dirname "$0")"
export REPS=2
OT="blk\\.(33|34|35|36|37)\\.ffn_.*_exps\\.weight=Vulkan1" ./sweep.sh amd ot2_wx5 "33" 48/0
OT="blk\\.(34|35|36|37|38)\\.ffn_.*_exps\\.weight=Vulkan1" NCMOE=34 UB=512 TS=48/0 REPS=3 TMO=3000 ./bench.sh final2_ot_wx5_n34_ub512 amd >/dev/null 2>&1; echo "final2_ot rc=$?"
echo Q16DONE
