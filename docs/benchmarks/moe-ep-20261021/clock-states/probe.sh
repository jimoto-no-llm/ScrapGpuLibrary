#!/bin/bash
# usage: probe.sh TAG [SPLIT]   one C1 (SPLIT "") or SGLX measurement under run.sh, with sampler.py running beside it.
# env REPS (default 5). Does not touch perf level (the caller sets/restores it). Output: $LOGS/cs_TAG.{log,out,csv}
SG=$HOME/moe-ep/sglx; . $SG/env.sh
TAG=$1; SPLIT=${2-}
python3 $SG/sampler.py $LOGS/cs_$TAG.csv & SP=$!
PROFILE=0 REPS=${REPS:-5} $SG/bench.sh cs_$TAG "$SPLIT" >/dev/null 2>&1
kill $SP 2>/dev/null; wait $SP 2>/dev/null
python3 - $LOGS/cs_$TAG.out <<P
import json,sys
r={}
for l in open(sys.argv[1]):
    try:d=json.loads(l)
    except: continue
    r[(d["n_prompt"],d["n_gen"])]=d
def g(k): return "%.1f"%r[k]["avg_ts"] if k in r else "-"
print("RESULT $TAG pp512=%s tg128=%s pg=%s  tg_samples=%s"%(g((512,0)),g((0,128)),g((512,128)),[round(x,1) for x in r[(0,128)]["samples_ts"]] if (0,128) in r else ""))
P
