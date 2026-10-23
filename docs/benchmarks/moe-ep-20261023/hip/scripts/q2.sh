#!/bin/bash
cd ~/moe-ep/hip
while [ ! -f logs/q1.done ]; do sleep 10; done
HB=$HOME/moe-ep/hip/llama-build/bin
. ~/moe-ep/hip/env.sh
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json
TMO=1500 ~/moe-ep/sglx/run.sh hip_tbo_rocm $HB/test-backend-ops test -b ROCm0 -o MUL_MAT,MUL_MAT_ID,GET_ROWS
TMO=1500 ./hkld.sh hip_kld_rocm_nc7 hip 7
TMO=1500 ./hkld.sh hip_kld_vk_nc7 vk 7
TMO=1500 ~/moe-ep/sglx/run.sh hip_perf_mmid_rocm $HB/test-backend-ops perf -b ROCm0 -o MUL_MAT_ID
TMO=1500 ~/moe-ep/sglx/run.sh hip_perf_mmid_vk $HB/test-backend-ops perf -b Vulkan0 -o MUL_MAT_ID
echo q2 done > logs/q2.done
