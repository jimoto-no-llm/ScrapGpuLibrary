#!/bin/bash
# C1 vs SGLX C(CL11) vs SGLX C(CUDA) under perf pin, alternating, REPS=10, PROFILE=0; restores perf auto.
. $HOME/moe-ep/sglx/env.sh
CPLUG=$HOME/moe-ep/cuda/build-plugin/libggml-cl11.so
LOGS=$HOME/moe-ep/sglx/logs
cleanup() { sudo -n /usr/local/sbin/sgl-driver-switch perf auto >> $LOGS/cuda_sglx.perf.log 2>&1; }
trap cleanup EXIT
b() { # TAG MODE(c1|cl11|cuda)
  local TAG=$1 MODE=$2
  sglx_env ""
  local DEVS=Vulkan0/Vulkan1 EXTRA=(-ts 5/1)
  if [ $MODE = cl11 ]; then sglx_env "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"; DEVS=Vulkan0/Vulkan1/SGLX0; EXTRA=(-ts 5/1/0 -ot 'token_embd\.weight=SGLX0'); fi
  if [ $MODE = cuda ]; then sglx_env "Vulkan0:0.5,Vulkan1:0.2,CUDA0:0.2,CUDA1:0.1"; export GGML_BACKEND_PATH=$CPLUG SCRAPGPU_CUDA_DEVICES="GT 730,GT 710"; DEVS=Vulkan0/Vulkan1/SGLX0; EXTRA=(-ts 5/1/0 -ot 'token_embd\.weight=SGLX0'); fi
  $RUN "$TAG" "$LB/llama-bench" -m "$MODEL" -dev "$DEVS" -ngl 99 -fa 0 -t 2 -b 512 -ub 512 -r "${REPS:-10}" -sm layer -ncmoe 0 "${EXTRA[@]}" -o jsonl -p 512 -n 128 -pg 512,128
}
export PROFILE=0 REPS=10
sudo -n /usr/local/sbin/sgl-driver-switch perf pin >> $LOGS/cuda_sglx.perf.log 2>&1
for i in 1 2; do
  b cuda_sglx_c1_$i c1
  b cuda_sglx_cl11_$i cl11
  b cuda_sglx_cuda_$i cuda
done
echo DONE >> $LOGS/cuda_sglx.perf.log
