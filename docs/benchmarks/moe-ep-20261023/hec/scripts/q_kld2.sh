. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
# the prefill-path KLD runs that failed in q_kld1 (llama-perplexity takes -dev A,B with a comma; hec_gen and llama-bench take A/B)
KL=$HOME/moe-ep/kl; TEXT=$KL/text.txt; PERP=$LB/llama-perplexity
COMMON=(-f $TEXT -c 512 --chunks 2 -b 512 -ub 512 -t 2 -fa off -sm layer)
export HEC_GEN_VERBOSE=1
P_ARGS=(-dev Vulkan0,SGLX0 -ts 1/0 -ngl 99 -ot "$OT" --poll 0)
kld() { tag=$1; shift; $RUN $LOGD kld_$tag $PERP -m $MODEL "${COMMON[@]}" "$@" --kl-divergence-base $KL/base30.kld --kl-divergence; }
kld ot_wx5_n34 -dev Vulkan0,Vulkan1 -ngl 99 -ncmoe 34 -ts 48/0 -ot 'blk\.(34|35|36|37|38)\.ffn_.*_exps\.weight=Vulkan1'
run_n $LOGD kld_sglx_rx $PERP -m $MODEL "${COMMON[@]}" "${P_ARGS[@]}" --kl-divergence-base $KL/base30.kld --kl-divergence
N_SPEC=${N_SPEC_MULTI:-Vulkan0:auto:400,CL11_0:300,Vulkan1:700:100,CL11_1:700} run_n $LOGD kld_sglx_multi $PERP -m $MODEL "${COMMON[@]}" "${P_ARGS[@]}" --kl-divergence-base $KL/base30.kld --kl-divergence
# decode path, the noise floor: everything cold (the ggml-cpu kernel only), and the GPU baseline against the SGLX one-device run
D="--text $KL/text.txt --n-tokens 450 --n-prefill 150 -ub 512"
N_SPEC=1 run_n $LOGD kld_dec_sglx_cold $GEN -m $MODEL "${N_ARGS[@]}" $D --dump $LOGD/dec_sglx_cold.bin
$GEN --compare $LOGD/dec_cpu.bin $LOGD/dec_sglx_cold.bin > $LOGD/kld_dec_cmp_sglx_cold.out
$GEN --compare $LOGD/dec_rx_n39.bin $LOGD/dec_sglx_rx.bin > $LOGD/kld_dec_cmp_rx39_vs_sglx_rx.out
