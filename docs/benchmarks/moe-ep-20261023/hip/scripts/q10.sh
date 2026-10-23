#!/bin/bash
cd ~/moe-ep/hip
export BDIR=llama-build-mmq
. ./env.sh
TMO=1500 ./hkld.sh mmq_kld_rocm_nc7 hip 7
TS=4/1 TMO=1500 ./hkld.sh mmq_kld_mix_nc0_ts41 mix 0
TMO=900 ./hbench.sh mmq_rx_nc7_ub512 hip 7 512 3
TMO=900 ./hbench.sh mmq_rx_nc7_ub128 hip 7 128 3
TMO=900 ./hbench.sh mmq_mix_nc0_ub512 mix 0 512 5
TMO=1500 ~/moe-ep/sglx/run.sh mmq_tbo_mmid $HOME/moe-ep/hip/llama-build-mmq/bin/test-backend-ops test -b ROCm0 -o MUL_MAT_ID
echo q10 done > logs/q10.done
