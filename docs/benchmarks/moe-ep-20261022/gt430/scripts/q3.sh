# q3 (under perf pin): the profile of C+430, CL11 server and CUDA server, and the same split without the GT 430 share for reference
export UB=320 REPS=3 PROFILE=1 PROFILE_EVERY=0
SPL="Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07"
for be in opencl11 cuda; do
  REMOTE_BE=$be ./bench.sh q3_C430_$be "$SPL" -p 512 -n 128 -pg 512,128
done
./bench.sh q3_C "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1" -p 512 -n 128 -pg 512,128
