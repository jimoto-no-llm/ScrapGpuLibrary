#!/bin/bash
# self-consistency at -ub 320: SGLX 1 device (save), then C vs that file
cd ~/moe-ep/fix/scripts
export KB=512 KUB=320
./kld.sh f2_one320 "Vulkan0:1" - save
./kld.sh f3_Cself320 "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1" $HOME/moe-ep/fix/kl/f2_one320.kld
