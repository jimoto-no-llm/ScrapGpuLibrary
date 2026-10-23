#!/bin/bash
# KLD against the CPU logits, and the self-consistency of SGLX (1 device vs N devices); perf auto
cd "$(dirname "$0")"
rm -f logs/queue6.done
A="Vulkan0:0.5,Vulkan1:0.5"; C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
./kld.sh kld_c1 "" cpu
./kld.sh kld_sglx1_vk0 "Vulkan0:1" cpu
./kld.sh kld_sglx1_vk0_save "Vulkan0:1" cpu save
./kld.sh kld_sglxA "$A" cpu
./kld.sh kld_sglxC "$C" cpu
./kld.sh kld_selfA_vs_1 "$A" "$HOME/moe-ep/sglx/kl/kld_sglx1_vk0_save.kld"
./kld.sh kld_selfC_vs_1 "$C" "$HOME/moe-ep/sglx/kl/kld_sglx1_vk0_save.kld"
echo done > logs/queue6.done
