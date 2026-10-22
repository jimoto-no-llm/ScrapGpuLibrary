. ./env.sh
Cs="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
Cc="Vulkan0:0.5,Vulkan1:0.2,CUDA0:0.2,CUDA1:0.1"
NEW=$HOME/moe-ep/fix/build/libggml-cl11.so
CUDA=$HOME/moe-ep/fix/build-cuda/libggml-cl11.so
export REPS=10 UB=320
for i in 1 2; do
  ./bench.sh f6_c1_ub320_$i ""
  PLUG=$NEW ./bench.sh f6_C_cl11_ub320_$i "$Cs"
  PLUG=$CUDA SCRAPGPU_CUDA_DEVICES="GT 730,GT 710" ./bench.sh f6_C_cuda_ub320_$i "$Cc"
done
export REPS=3
for ub in 96 128 160 192 256; do UB=$ub B=256 ./bench.sh f7_C_p256_ub$ub "$Cs" -p 256 -n 0; done
for ub in 256 320 384 448 512; do UB=$ub B=1024 ./bench.sh f7_C_p1024_ub$ub "$Cs" -p 1024 -n 0; done
UB=256 B=256 ./bench.sh f7_c1_p256_ub256 "" -p 256 -n 0
UB=128 B=256 ./bench.sh f7_c1_p256_ub128 "" -p 256 -n 0
UB=512 B=1024 ./bench.sh f7_c1_p1024_ub512 "" -p 1024 -n 0
UB=320 B=1024 ./bench.sh f7_c1_p1024_ub320 "" -p 1024 -n 0
UB=320 REPS=3 ./bench.sh f6_refend_C "$Cs"
