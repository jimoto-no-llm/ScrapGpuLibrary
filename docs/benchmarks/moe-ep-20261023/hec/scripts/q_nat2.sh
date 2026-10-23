. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
export HEC_GEN_VERBOSE=1
P=$HOME/moe-ep/hec/prompts
# the same prompts with a prior of 64 tokens of selections from the other texts (prior_nat.txt)
for pr in nat_en nat_ja nat_code nat_chat; do
  ( export GGML_SGLX_MOE_STATS_IN=$LOGD/prior_nat.txt GGML_SGLX_MOE_PRIOR_TOKENS=${PRIOR_TOKENS:-64}; run_n $LOGD txtNp_$pr $GEN -m $MODEL -ub 512 "${N_ARGS[@]}" --text $P/$pr.txt --n-prompt 1000 --n-gen 128 )
done
