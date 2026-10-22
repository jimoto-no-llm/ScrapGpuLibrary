#!/bin/bash
cd "$(dirname "$0")"
# tg32 vs tg128 on the same state (RX n39): is the sweep value (tg32 after pp128) or the final value (tg128) representative?
for n in 32 128 32 128; do
  NCMOE=39 UB=512 REPS=3 TMO=900 ./bench.sh chk_rx_n39_tg$n rx -p 0 -n $n >/dev/null 2>&1
  python3 - logs/chk_rx_n39_tg$n.out chk_rx_n39_tg$n <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l); print(sys.argv[2], [round(x,2) for x in d["samples_ts"]])
P
done
echo Q11DONE
