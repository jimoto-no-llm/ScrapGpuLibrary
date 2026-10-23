#!/bin/bash
cd ~/moe-ep/hip
while [ ! -f logs/q5.done ]; do sleep 10; done
. ./env.sh
TMO=300 ~/moe-ep/sglx/run.sh hip_bw ./bw
echo q6 done > logs/q6.done
