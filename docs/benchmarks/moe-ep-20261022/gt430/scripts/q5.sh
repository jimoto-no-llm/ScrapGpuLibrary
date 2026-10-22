# q5 (under perf pin): the autotuned split of C+430 (3 rounds), CL11 server
export UB=320 REPS=3 PROFILE=1 PROFILE_EVERY=0
SPL="Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07"
AUTOTUNE=3,24 REMOTE_BE=opencl11 ./bench.sh q5_C430_auto_opencl11 "$SPL" -p 512 -n 256 -pg 512,128
AUTOTUNE=3,24 ./bench.sh q5_C_auto "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1" -p 512 -n 256 -pg 512,128
