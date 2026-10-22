C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
SGK=$HOME/moe-ep/sglx/kl
./kld.sh pp3_one "Vulkan0:1" cpu save
./kld.sh pp3_Cself128 "$C" $SGK/pp3_one.kld
./kld.sh pp3_Aself128 "Vulkan0:0.5,Vulkan1:0.5" $SGK/pp3_one.kld
KB=512 KUB=320 ./kld.sh pp3_C320 "$C" cpu
KB=512 KUB=512 ./kld.sh pp3_C512 "$C" cpu
