#!/bin/bash
. $HOME/moe-ep/sglx/env.sh
sglx_env "Vulkan0:0.5,Vulkan1:0.2,CUDA0:0.2,CUDA1:0.1"
export GGML_BACKEND_PATH=$HOME/moe-ep/cuda/build-plugin/libggml-cl11.so SCRAPGPU_CUDA_DEVICES="GT 730,GT 710"
for spec in "CUDA0:1" "CUDA0:0.7,CUDA1:0.3" "Vulkan0:0.5,Vulkan1:0.2,CUDA0:0.2,CUDA1:0.1"; do
  t=$(echo "$spec" | tr -c 'A-Za-z0-9\n' '_')
  export GGML_SGLX_SPLIT="$spec"
  $RUN cuda_tbo_$t $LB/test-backend-ops test -b SGLX0 -o MUL_MAT,GET_ROWS
done
