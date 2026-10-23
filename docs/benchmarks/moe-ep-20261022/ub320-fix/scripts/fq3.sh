export REPS=10
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
OLD=$HOME/moe-ep/pp/build-cl11/libggml-cl11.so
export UB=320
for i in 1 2; do
./bench.sh f5_c1_ub320_$i ""
./bench.sh f5_C_new_ub320_$i "$C"
PLUG=$OLD ./bench.sh f5_C_old_ub320_$i "$C"
done
