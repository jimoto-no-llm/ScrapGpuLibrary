. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
# (1) LFM2.5-8B-A1B: the procedure of docs/benchmarks/moe-ep-20261022/ub320-fix (text.txt, base.kld = CPU logits, -c 512 --chunks 4 -b 512 -ub 320).
#     The earlier lfmmoe_cmp_* (KLD 0.49) compared hec_gen dumps of RANDOM tokens, which is out of distribution for a small model: not comparable.
KL=$HOME/moe-ep/kl; PERP=$LB/llama-perplexity
LM=$HOME/moe-ep/models/LFM2.5-8B-A1B-Q4_0.gguf
LC=(-f $KL/text.txt -c 512 --chunks 4 -b 512 -ub 320 -t 2 -fa off -sm layer)
export HEC_GEN_VERBOSE=1
P_ARGS=(-dev Vulkan0,SGLX0 -ts 1/0 -ngl 99 -ot "$OT" --poll 0)
$RUN $LOGD kld3_lfm_c1 $PERP -m $LM "${LC[@]}" -dev Vulkan0,Vulkan1 -ngl 99 -ts 5/1 --kl-divergence-base $KL/base.kld --kl-divergence
( export N_SPEC=Vulkan0:auto; run_n $LOGD kld3_lfm_moe_rx $PERP -m $LM "${LC[@]}" "${P_ARGS[@]}" --kl-divergence-base $KL/base.kld --kl-divergence )
( export N_SPEC=Vulkan0:auto,CL11_0:200; run_n $LOGD kld3_lfm_moe_rx_gt730 $PERP -m $LM "${LC[@]}" "${P_ARGS[@]}" --kl-divergence-base $KL/base.kld --kl-divergence )
# (2) 30B with 6 chunks (-c 512 -b 512 -ub 512): a new CPU base, then rx_n39, N, the 4-device configuration
B6=$KL/base30_6.kld
C6=(-f $KL/text.txt -c 512 --chunks 6 -b 512 -ub 512 -t 2 -fa off -sm layer)
[ -f $B6 ] || $RUN $LOGD kld3_cpu_base6 $PERP -m $MODEL "${C6[@]}" -dev none -ngl 0 --kl-divergence-base $B6
k6() { tag=$1; shift; $RUN $LOGD kld3_$tag $PERP -m $MODEL "${C6[@]}" "$@" --kl-divergence-base $B6 --kl-divergence; }
k6 rx_n39 -dev Vulkan0 -ngl 99 -ncmoe 39
run_n $LOGD kld3_N $PERP -m $MODEL "${C6[@]}" "${P_ARGS[@]}" --kl-divergence-base $B6 --kl-divergence
k6 rx_n41 -dev Vulkan0 -ngl 99 -ncmoe 41
N_SPEC=Vulkan0:auto:400,CL11_0:300,Vulkan1:700:100,CL11_1:700 run_n $LOGD kld3_multi $PERP -m $MODEL "${C6[@]}" "${P_ARGS[@]}" --kl-divergence-base $B6 --kl-divergence
