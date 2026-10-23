# the SGLX MoE configuration under test ("N"): sourced by the final queues
PLUG=$H/plug/libggml-cl11.so
OT='blk\.[0-9]+\.ffn_(up|gate|down)_exps\.weight=SGLX0'
n_set_env() {   # environment of config N (call in a subshell)
  export GGML_BACKEND_PATH=$PLUG GGML_CL11_PLATFORMS=NVIDIA GGML_CL11_DEVICES="GT 730,GT 710" GGML_CL11_AUTO_RPC=0
  export GGML_SGLX_MOE=${N_SPEC:-Vulkan0:auto:400,CL11_0:300} GGML_SGLX_MOE_THREADS=${N_THREADS:-2} GGML_SGLX_MOE_SPIN_US=${N_SPIN:-0}
  export GGML_SGLX_MOE_VERBOSE=1 GGML_SGLX_PROFILE=1
}
run_n() { ( n_set_env; $RUN "$@" ); }     # run_n LOGDIR TAG cmd...   (N_SPEC etc. in the environment select the variant)
N_ARGS=(-dev Vulkan0/SGLX0 -ts 1/0 -ngl 99 -ot "$OT" --poll 0)
