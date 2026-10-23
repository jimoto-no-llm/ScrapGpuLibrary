#!/bin/bash
cd ~/moe-ep/hip
while [ ! -f logs/q10.done ]; do sleep 10; done
sudo -n /usr/local/sbin/sgl-driver-switch status | grep perf > logs/perf_status_q11.txt
TMO=900 ./hbench.sh ctl_vk_rx_nc7_ub512 vk 7 512 3
TMO=900 ./hbench.sh ctl_hip_rx_nc7_ub512 hip 7 512 3
TMO=900 ./hbench.sh ctl_vkmix_nc0_ub512 vkmix 0 512 5
TMO=900 ./hbench.sh ctl_hipmix_nc0_ub512 mix 0 512 5
sudo -n /usr/local/sbin/sgl-driver-switch status | grep perf >> logs/perf_status_q11.txt
echo q11 done > logs/q11.done
