# q9 (under perf pin): the scheduling mode of the CUDA context of the server (the host has 2 cores)
export UB=320 REPS=3 PROFILE=1 PROFILE_EVERY=0 REMOTE_BE=cuda
SPL="Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07"
for m in blocking yield spin; do
  CUDA_SCHED=$m ./bench.sh q9_cuda_$m "$SPL" -p 512 -n 128 -pg 512,128
done
