#!/bin/bash
cd ~/moe-ep/hip
while [ ! -f logs/q7.done ]; do sleep 10; done
BDIR=llama-build-mmq HIP_EXTRA=-DGGML_CUDA_FORCE_MMQ=ON ./build-hip.sh
echo q8 done > logs/q8.done
