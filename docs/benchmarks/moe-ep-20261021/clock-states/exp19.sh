#!/bin/bash
# exp19: can perf writes change the state? Cases (each: bench C1 tg-only REPS=6; features = number of enabled gfx features in pp_features of card1 at +10 s and +24 s):
#   A high | B high,high | C high + mid-run (+12 s) high again | D high + mid-run auto,high | E (no initial write) mid-run high only
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
SW="sudo -n /usr/local/sbin/sgl-driver-switch perf"
export REPS=6 PROFILE=0
feat() { grep -E "DPM_GFXCLK|GFXOFF|DS_GFXCLK" /sys/class/drm/card1/device/pp_features | awk '{printf "%s=%s ", $2, $NF}'; }
one() { # one TAG initial-writes midrun-writes
  T=e19_$1; init=$2; mid=$3
  for m in $init; do $SW $m >> logs/cs.perf.log 2>&1; done
  $SG/bench.sh cs_$T "" -p 0 -n 128 >/dev/null 2>&1 & BP=$!
  sleep 10; f1=$(feat)
  if [ -n "$mid" ]; then sleep 2; for m in $mid; do $SW $m >> logs/cs.perf.log 2>&1; done; fi
  sleep 3; f2=$(feat); sleep 8; f3=$(feat)
  wait $BP
  v=$(python3 - $SG/logs/cs_$T.out <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("%.1f"%d["avg_ts"], [round(x,1) for x in d["samples_ts"]])
P
)
  echo "$(date +%T) $1 init=[$init] mid=[$mid] tg=$v | +10s: $f1| +15s: $f2| +23s: $f3"
  $SW auto >> logs/cs.perf.log 2>&1; sleep 20
}
for r in 1 2; do
one A$r "auto high" ""
one B$r "auto high high" ""
one C$r "auto high" "high"
one D$r "auto high" "auto high"
one E$r "auto" "high"
done
echo FIN
