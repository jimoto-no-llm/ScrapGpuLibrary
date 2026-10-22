C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
SGK=$HOME/moe-ep/sglx/kl
./kld.sh pp_one "Vulkan0:1" cpu save
./kld.sh pp_C128 "$C" cpu
./kld.sh pp_Cself128 "$C" $SGK/pp_one.kld
KB=512 KUB=320 ./kld.sh pp_C320 "$C" cpu
KB=512 KUB=320 ./kld.sh pp_Cself320 "$C" $SGK/pp_one.kld
./kld.sh pp_c1_128 "" cpu
