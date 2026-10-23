#!/bin/bash
cd ~/moe-ep/hip
while [ ! -f logs/q2.done ]; do sleep 10; done
TMO=900 ./hbench.sh vkold_rx_nc7_ub512 vkold 7 512 3
TMO=900 ./hbench.sh vk_rx_nc8_ub512_b vk 8 512 3
GGML_CUDA_FORCE_MMQ=1 TMO=900 ./hbench.sh hip_rx_nc7_ub512_mmq hip 7 512 3
GGML_CUDA_FORCE_CUBLAS=1 TMO=900 ./hbench.sh hip_rx_nc7_ub512_cublas hip 7 512 3
TMO=900 ./hbench.sh hip_rx_nc7_ub128 hip 7 128 3
TMO=900 ./hbench.sh vk_rx_nc7_ub128 vk 7 128 3
TMO=900 ./hbench.sh vkoldmix_nc0_ub512 vkoldmix 0 512 3
TMO=900 ./hbench.sh vkmix_nc0_ub512 vkmix 0 512 3
TMO=900 ./hbench.sh hipmix_nc0_ub512 mix 0 512 3
echo q3 done > logs/q3.done
