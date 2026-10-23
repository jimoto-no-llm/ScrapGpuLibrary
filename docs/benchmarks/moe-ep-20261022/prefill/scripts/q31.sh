C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
SGK=$HOME/moe-ep/sglx/kl
./kld.sh pp2_C128 "$C" cpu
./kld.sh pp2_Cself128 "$C" $SGK/pp_one.kld
KB=512 KUB=192 ./kld.sh pp2_C192 "$C" cpu
KB=512 KUB=256 ./kld.sh pp2_C256 "$C" cpu
