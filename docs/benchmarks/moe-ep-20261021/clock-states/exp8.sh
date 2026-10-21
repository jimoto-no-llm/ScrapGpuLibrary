#!/bin/bash
# exp8: which sysfs reads change the C1 tg state? Under perf high, C1 REPS=3: bare, then the sampler (20 ms) restricted to one column group, then bare again.
SG=$HOME/moe-ep/sglx; cd $SG
trap "sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> logs/cs.perf.log 2>&1" EXIT
sudo -n /usr/local/sbin/sgl-driver-switch perf high >> logs/cs.perf.log 2>&1
export REPS=3 PROFILE=0
one() { # one TAG [COLS-regex]
  if [ -n "$2" ]; then SAMPLE_COLS="$2" $SG/probe.sh $1 ""; else
    $SG/bench.sh cs_$1 "" >/dev/null 2>&1
    python3 - $SG/logs/cs_$1.out $1 <<P
import json,sys
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d["n_gen"]==128 and d["n_prompt"]==0: print("RESULT",sys.argv[2],"tg128=%.1f"%d["avg_ts"],flush=True)
P
  fi
}
one e8_bare1; one e8_bare2
for g in "_sclk$" "_busy$" "_temp$|_pwr$" "cpu" "_lspeed|_lwidth|_pcie" "_mclk|_fclk|_socclk"; do
  one "e8_$(echo $g | tr -c 'a-z0-9\n' _)" "$g"; one e8_bare_after
done
echo FIN
