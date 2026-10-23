. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
# correctness (docs/plans/HOT_EXPERT_CACHE.md section 8): KL divergence and top-1 against the CPU logits of the same model
KL=$HOME/moe-ep/kl; TEXT=$KL/text.txt; PERP=$LB/llama-perplexity
COMMON=(-f $TEXT -c 512 --chunks 2 -b 512 -ub 512 -t 2 -fa off -sm layer)
export HEC_GEN_VERBOSE=1
# 1. prefill path (llama-perplexity, N = 512 per pass): the CPU base, then the configurations
[ -f $KL/base30.kld ] || $RUN $LOGD kld_cpu_base $PERP -m $MODEL "${COMMON[@]}" -dev none -ngl 0 --kl-divergence-base $KL/base30.kld
kld() { tag=$1; shift; $RUN $LOGD kld_$tag $PERP -m $MODEL "${COMMON[@]}" "$@" --kl-divergence-base $KL/base30.kld --kl-divergence; }
kld rx_n39 -dev Vulkan0 -ngl 99 -ncmoe 39
kld ot_wx5_n34 -dev Vulkan0/Vulkan1 -ngl 99 -ncmoe 34 -ts 48/0 -ot 'blk\.(34|35|36|37|38)\.ffn_.*_exps\.weight=Vulkan1'
run_n $LOGD kld_sglx_rx $PERP -m $MODEL "${COMMON[@]}" "${N_ARGS[@]}" --kl-divergence-base $KL/base30.kld --kl-divergence
NM=${N_SPEC_MULTI:-Vulkan0:auto:250,CL11_0:300,Vulkan1:700:100,CL11_1:700}
N_SPEC=$NM run_n $LOGD kld_sglx_multi $PERP -m $MODEL "${COMMON[@]}" "${N_ARGS[@]}" --kl-divergence-base $KL/base30.kld --kl-divergence
# 2. decode path (N = 1 after a prompt of 150 tokens): teacher forcing on the same text, logits written to files and compared
D="--text $KL/text.txt --n-tokens 450 --n-prefill 150 -ub 512"
$RUN $LOGD kld_dec_cpu $GEN -m $MODEL -dev none -ngl 0 $D --dump $LOGD/dec_cpu.bin
$RUN $LOGD kld_dec_rx_n39 $GEN -m $MODEL -dev Vulkan0 -ngl 99 -ncmoe 39 $D --dump $LOGD/dec_rx_n39.bin
run_n $LOGD kld_dec_sglx_rx $GEN -m $MODEL "${N_ARGS[@]}" $D --dump $LOGD/dec_sglx_rx.bin
N_SPEC=$NM run_n $LOGD kld_dec_sglx_multi $GEN -m $MODEL "${N_ARGS[@]}" $D --dump $LOGD/dec_sglx_multi.bin
for f in rx_n39 sglx_rx sglx_multi; do $GEN --compare $LOGD/dec_cpu.bin $LOGD/dec_$f.bin > $LOGD/kld_dec_cmp_$f.out; done
$GEN --compare $LOGD/dec_sglx_rx.bin $LOGD/dec_sglx_multi.bin > $LOGD/kld_dec_cmp_self.out
