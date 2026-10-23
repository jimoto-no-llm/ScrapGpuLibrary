# q10 (under perf pin): the comparison. C1 / SGLX C / SGLX C+430 (A, G) / G without the 430 (H) / A autotuned, alternating, twice; REPS=10, PROFILE=0, -ub 320
export UB=320 REPS=10 PROFILE=0 REMOTE_BE=opencl11
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
A="Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07"
B="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,REMOTE0:0.1"
D="Vulkan0:0.45,Vulkan1:0.15,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.12"
G="Vulkan0:0.6,CL11_0:0.25,CL11_1:0.05,REMOTE0:0.1"
H="Vulkan0:0.65,CL11_0:0.28,CL11_1:0.07"
for i in 1 2; do
  ./bench.sh q10_c1_$i ""
  ./bench.sh q10_C_$i "$C"
  ./bench.sh q10_A_$i "$A"
  ./bench.sh q10_G_$i "$G"
  ./bench.sh q10_H_$i "$H"
  PLUG=$HOME/moe-ep/g430/plug3/libggml-cl11.so AUTOTUNE=2,16 ./bench.sh q10_Aauto_$i "$A"
done
