# q6 (under perf pin): one round trip per job (batching) against three, and the CUDA server
export UB=320 REPS=3 PROFILE=1 PROFILE_EVERY=0
SPL="Vulkan0:0.45,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.08,REMOTE0:0.07"
RBATCH=1 REMOTE_BE=opencl11 ./bench.sh q6_b1_opencl11 "$SPL" -p 512 -n 128 -pg 512,128
RBATCH=0 REMOTE_BE=opencl11 ./bench.sh q6_b0_opencl11 "$SPL" -p 512 -n 128 -pg 512,128
RBATCH=1 REMOTE_BE=cuda ./bench.sh q6_b1_cuda "$SPL" -p 512 -n 128 -pg 512,128
# the job alone, device idle
. ./env.sh
B=$ROOT/g430/srv2/scrapgpu_remote_bench
for be in opencl11 cuda; do for bt in "" --batch; do
  SCRAPGPU_DEVICE_SERVER=$SERVER SCRAPGPU_REMOTE_BACKEND=$be GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 430" SCRAPGPU_CUDA_DEVICES="GT 430" \
    ./run.sh q6_rbench_${be}${bt} $B --rows 4608 --k 2048 --iters 200 $bt
done; done
