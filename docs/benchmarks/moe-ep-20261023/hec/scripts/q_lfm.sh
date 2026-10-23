. $HOME/moe-ep/hec/env_hec.sh
# LFM2.5-8B-A1B regression of the SGLX output-projection split (config H of docs/benchmarks/moe-ep-20261023/gt430-latency: tg 69.0 / pg 240.4 with perf pin), new plugin build, MoE mode off
LM=$HOME/moe-ep/models/LFM2.5-8B-A1B-Q4_0.gguf
PLUG=$H/plug/libggml-cl11.so
H_ENV="GGML_SGLX_SPLIT=Vulkan0:0.78,CL11_0:0.17,CL11_1:0.05 GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES=GT_730,GT_710 GGML_CL11_AUTO_RPC=0 GGML_SGLX_PROFILE=0"
for round in 1 2; do
  $RUN $LOGD lfm_c1_$round taskset -c 0 $LB/llama-bench -m $LM -dev Vulkan0/Vulkan1 -ngl 99 -fa 0 -t 2 -b 512 -ub 320 -r 5 -sm layer -ncmoe 0 -ts 5/1 -o jsonl -p 512 -n 128 -pg 512,128
  env GGML_SGLX_SPLIT=Vulkan0:0.78,CL11_0:0.17,CL11_1:0.05 GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0 GGML_SGLX_PROFILE=0 \
    $RUN $LOGD lfm_H_$round taskset -c 0 $LB/llama-bench -m $LM -dev Vulkan0/Vulkan1/SGLX0 -ngl 99 -fa 0 -t 2 -b 512 -ub 320 -r 5 -sm layer -ncmoe 0 -ts 5/1/0 -ot 'token_embd\.weight=SGLX0' -o jsonl -p 512 -n 128 -pg 512,128
done
