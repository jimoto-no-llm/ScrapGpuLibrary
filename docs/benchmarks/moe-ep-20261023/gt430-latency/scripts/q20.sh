# q20: job time of the GT 430 (CL11 / CUDA server) alone: back-to-back, with a 12 ms gap (token generation), with 2 spinning threads (llama-bench -t 2)
cd ~/moe-ep/g430b/scripts
. ./env.sh
B=$ROOT/g430b/srv/scrapgpu_remote_bench
for be in opencl11 cuda; do
  export SCRAPGPU_DEVICE_SERVER=$SERVER SCRAPGPU_REMOTE_BACKEND=$be GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 430" SCRAPGPU_CUDA_DEVICES="GT 430"
  ./run.sh q20_${be}_g0_h0 $B --gap-us 0 --hog 0 --iters 400
  ./run.sh q20_${be}_g12_h0 $B --gap-us 12000 --hog 0 --iters 400
  ./run.sh q20_${be}_g12_h2 $B --gap-us 12000 --hog 2 --iters 400
  ./run.sh q20_${be}_g12_h2_b $B --gap-us 12000 --hog 2 --iters 400 --batch
  ./run.sh q20_${be}_g0_h2 $B --gap-us 0 --hog 2 --iters 400
  ./run.sh q20_${be}_g12spin_h0 $B --gap-us 12000 --spin-gap --hog 0 --iters 400
done
echo Q20_DONE >> ../logs/queue.log
