. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
export HEC_GEN_VERBOSE=1
# generality: LFM2.5-8B-A1B (sigmoid router with a bias, 32 experts of 2048 x 1792, 4 used, dense first layers) through the MoE mode; the logits against the CPU
LM=$HOME/moe-ep/models/LFM2.5-8B-A1B-Q4_0.gguf
D="--n-tokens 200 --n-prefill 100 -ub 512"
$RUN $LOGD lfmmoe_cpu $GEN -m $LM -dev none -ngl 0 $D --dump $LOGD/lfm_cpu.bin
$RUN $LOGD lfmmoe_vk $GEN -m $LM -dev Vulkan0 -ngl 99 -ncmoe 7 $D --dump $LOGD/lfm_vk.bin
( export N_SPEC=Vulkan0:auto; run_n $LOGD lfmmoe_sglx $GEN -m $LM "${N_ARGS[@]}" $D --dump $LOGD/lfm_sglx.bin )
$GEN --compare $LOGD/lfm_cpu.bin $LOGD/lfm_vk.bin > $LOGD/lfmmoe_cmp_vk.out
$GEN --compare $LOGD/lfm_cpu.bin $LOGD/lfm_sglx.bin > $LOGD/lfmmoe_cmp_sglx.out
run_n $LOGD lfmmoe_bench $GEN -m $LM -ub 512 "${N_ARGS[@]}" -r 2 --bench pp512,tg128
