#!/bin/bash
cd ~/moe-ep/hip
while [ ! -f logs/q4.done ]; do sleep 10; done
TMO=900 ./hbench.sh hip_rx_nc7_ub512_def2 hip 7 512 3
GGML_CUDA_FORCE_MMQ=1 TMO=900 ./hbench.sh hip_rx_nc7_ub512_mmq2 hip 7 512 3
TMO=900 ./hbench.sh hip_rx_nc7_ub512_def3 hip 7 512 3
TMO=900 ./hbench.sh vk_rx_nc7_ub512_c vk 7 512 3
echo q5 done > logs/q5.done
