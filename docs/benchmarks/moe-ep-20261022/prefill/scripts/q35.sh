C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
PLUG_OVERRIDE=1 true
sed "s#^PLUG=.*#PLUG=\$ROOT/sglx/build-cl11/libggml-cl11.so#" env.sh > env_old.sh
sed "s#/env.sh#/env_old.sh#" kld.sh > kld_old.sh; chmod +x kld_old.sh
KB=512 KUB=320 ./kld_old.sh pp6_old320 "$C" cpu
