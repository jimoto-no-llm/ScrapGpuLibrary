#!/bin/bash
cd ~/moe-ep/hip
while [ ! -f logs/q6.done ]; do sleep 10; done
TMO=900 ./hbench.sh hipmix_nc0_ub512_b mix 0 512 5
TMO=900 ./hbench.sh vkmix_nc0_ub512_b vkmix 0 512 5
TMO=1500 ./hkld.sh hip_kld_mix_nc0 mix 0
TMO=1500 ./hkld.sh hip_kld_vkmix_nc0 vkmix 0
echo q7 done > logs/q7.done
