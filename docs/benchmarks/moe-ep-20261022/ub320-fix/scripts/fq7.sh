cd ~/moe-ep/fix/scripts
Cs="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
for ub in 128 192 256 512; do KB=512 KUB=$ub ./kld.sh f8_C_ub$ub "$Cs" cpu; done
KB=128 KUB=128 ./kld.sh f8_C_ub128b128 "$Cs" cpu
KB=512 KUB=320 ./kld.sh f8_c1_ub320 "" cpu
