. $HOME/moe-ep/hec/env_hec.sh
. $HOME/moe-ep/hec/cfg_n.sh
export HEC_GEN_VERBOSE=1
P=$HOME/moe-ep/hec/prompts
# natural text: 128 greedy tokens after each prompt; the SGLX configuration N alone ("static prefill placement"), and with a prior for the frequencies made from other text (stats file)
for pr in nat_en nat_ja nat_code nat_chat; do
  run_n $LOGD txtN_$pr $GEN -m $MODEL -ub 512 "${N_ARGS[@]}" --text $P/$pr.txt --n-prompt 1000 --n-gen 128
done
