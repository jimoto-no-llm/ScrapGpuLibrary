#!/bin/bash
cd ~/moe-ep/hip
for nc in 6 7 8; do TMO=900 ./hbench.sh hip_rx_nc${nc}_ub512 hip $nc 512 3; done
for nc in 5 6 7 8; do TMO=900 ./hbench.sh vk_rx_nc${nc}_ub512 vk $nc 512 3; done
echo q1 done > logs/q1.done
