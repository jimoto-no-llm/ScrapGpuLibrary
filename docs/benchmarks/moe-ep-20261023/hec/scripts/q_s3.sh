. $HOME/moe-ep/hec/env_hec.sh
export HEC_GEN_VERBOSE=1
B=$LB
# MB-H5: how fast can experts go from the mmap into Vulkan0 slots (A per-expert sync, B staged+async, C two-thread pipeline, D host-pointer import)
$RUN $LOGD s3_stream_c12 $H/bin/hec_bench stream --model $MODEL --cpu-lib $B/libggml-cpu.so --vk-lib $B/libggml-vulkan.so --vk-dev Vulkan0 --chunk 12 --iters 12 --modes ABCD
