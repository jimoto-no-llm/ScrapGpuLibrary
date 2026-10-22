# q4 (under perf pin): where the time of one job goes on the GT 430, CL11 server and CUDA server
cd ~/moe-ep/g430/scripts
. ./env.sh
B=$ROOT/g430/srv/scrapgpu_remote_bench
for be in opencl11 cuda; do
  for rows in 4608 1024; do
    SCRAPGPU_DEVICE_SERVER=$SERVER SCRAPGPU_REMOTE_BACKEND=$be GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 430" SCRAPGPU_CUDA_DEVICES="GT 430" \
      ./run.sh q4_rbench_${be}_$rows $B --rows $rows --k 2048 --iters 200
  done
done
