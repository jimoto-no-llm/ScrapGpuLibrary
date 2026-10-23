# q8 (under perf pin): pilot of other splits with the GT 430 (REPS=3, profile on)
export UB=320 REPS=3 PROFILE=1 PROFILE_EVERY=0 REMOTE_BE=opencl11
for v in "B Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,REMOTE0:0.1" \
         "C Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.05,REMOTE0:0.05" \
         "D Vulkan0:0.45,Vulkan1:0.15,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.12" \
         "E Vulkan0:0.55,Vulkan1:0.15,CL11_0:0.2,REMOTE0:0.1" \
         "F Vulkan0:0.6,Vulkan1:0.15,CL11_0:0.15,REMOTE0:0.1" \
         "G Vulkan0:0.6,CL11_0:0.25,CL11_1:0.05,REMOTE0:0.1"; do
  set -- $v
  ./bench.sh q8_$1 "$2" -p 512 -n 128 -pg 512,128
done
