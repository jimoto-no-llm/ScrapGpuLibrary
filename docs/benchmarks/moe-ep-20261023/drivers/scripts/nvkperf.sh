#!/bin/bash
# nvkperf.sh LABEL : NVK mul_mat q4_0 (m=4096,n=1,k=14336, 33 MB weights) on both Kepler cards; idx0=GT 710, idx1=GT 730
cd ~/moe-ep/src/llama-build/bin; mkdir -p ~/moe-ep/drivers/logs/nvkperf
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nouveau_icd.json NVK_I_WANT_A_BROKEN_VULKAN_DRIVER=1
for i in 0 1; do
  { echo "# $1 idx=$i"; GGML_VK_VISIBLE_DEVICES=$i timeout 100 flock ~/bench/gpu-bench.lock ./test-backend-ops perf -b Vulkan0 -o MUL_MAT -p "type_a=q4_0,type_b=f32,m=4096,n=1,k=14336" 2>&1 | grep -E "Device desc|MUL_MAT"; } | cut -c1-200 | tee ~/moe-ep/drivers/logs/nvkperf/$1_idx$i.txt
done
