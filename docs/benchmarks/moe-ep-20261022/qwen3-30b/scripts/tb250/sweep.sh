#!/bin/bash
# usage: sweep.sh CFG PREFIX "ncmoe list..." [TS]    -r 1 -p 128 -n 32 per point, descending list; stops at the first failure (no result line for n_gen)
cd "$(dirname "$0")"
CFGN=$1; PFX=$2; LIST=$3; export TS=$4
for n in $LIST; do
  tag=sw_${PFX}_n${n}
  NCMOE=$n REPS=${REPS:-1} UB=${UB:-512} TMO=${TMO:-900} ./bench.sh $tag $CFGN -p 128 -n 32 >/dev/null 2>&1
  if ! grep -q "\"n_gen\": 32" logs/$tag.out 2>/dev/null; then echo "$tag FAIL"; break; fi
  python3 - logs/$tag.out $tag <<P
import json,sys
r={}
for l in open(sys.argv[1]):
    d=json.loads(l); r[(d["n_prompt"],d["n_gen"])]=d["avg_ts"]
print(sys.argv[2], "pp128=%.2f tg32=%.2f"%(r.get((128,0),0), r.get((0,32),0)))
P
done
