#!/bin/bash
cd ~/moe-ep/hip
TS=4/1 TMO=1500 ./hkld.sh hip_kld_mix_nc0_ts41 mix 0
TMO=1500 ./hkld.sh hip_kld_mix_nc0_retry mix 0
echo q9 done > logs/q9.done
