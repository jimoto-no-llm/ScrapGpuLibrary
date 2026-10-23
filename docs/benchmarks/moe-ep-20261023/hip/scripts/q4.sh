#!/bin/bash
cd ~/moe-ep/hip
while [ ! -f logs/q3.done ]; do sleep 10; done
GGML_CUDA_FORCE_MMQ=1 TMO=1500 ./hkld.sh hip_kld_rocm_nc7_mmq hip 7
GGML_CUDA_FORCE_CUBLAS=1 TMO=1500 ./hkld.sh hip_kld_rocm_nc7_cublas hip 7
echo q4 done > logs/q4.done
