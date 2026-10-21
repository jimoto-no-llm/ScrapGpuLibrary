#!/bin/bash
# exp9: bare C1 (no sampler, REPS=3) N times under perf high; before and after each run a snapshot of the active power profile and idle clocks of both cards. Restores perf auto.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
export REPS=3 PROFILE=0
snap() { for c in 1 5; do d=/sys/class/drm/card$c/device; printf "c$c prof[%s] sclk[%s] mclk[%s] " "$(grep -E '^ *[0-9]+ .*\*' $d/pp_power_profile_mode | head -1 | awk '{print $1,$2}' | tr -d '*')" "$(grep '\*' $d/pp_dpm_sclk | tr -d '\n')" "$(grep '\*' $d/pp_dpm_mclk | tr -d '\n')"; done; echo; }
for i in $(seq 1 ${1:-10}); do
  T=${2:-e9}_$i; echo "pre  $T $(snap)"
  $SG/bench.sh cs_$T "" >/dev/null 2>&1
  python3 - $SG/logs/cs_$T.out $T <<P
import json,sys,time
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("RESULT",time.strftime("%H:%M:%S"),sys.argv[2],"tg128=%.1f"%d["avg_ts"],[round(x,1) for x in d["samples_ts"]],flush=True)
P
  echo "post $T $(snap)"
done
echo FIN
