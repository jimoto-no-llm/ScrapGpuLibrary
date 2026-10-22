# q12 (under perf pin): without the RX/WX path. SGLX C / G (C+430 without the WX 2100) / H (G without the GT 430), alternating, twice; REPS=10, -ub 320
export UB=320 REPS=10 PROFILE=0 REMOTE_BE=opencl11
C="Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"
G="Vulkan0:0.6,CL11_0:0.25,CL11_1:0.05,REMOTE0:0.1"
H="Vulkan0:0.65,CL11_0:0.28,CL11_1:0.07"
for i in 1 2; do
  ./bench.sh q12_C_$i "$C"
  ./bench.sh q12_G_$i "$G"
  ./bench.sh q12_H_$i "$H"
done
./bench.sh q12_c1_1 ""
