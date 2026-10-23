#!/bin/bash
# usage: kld.sh TAG SPLIT BASE [save]
#   llama-perplexity -c 512 --chunks 4 -b ${KB:-128} -ub ${KUB:-128} -fa off on the text of the baseline (docs/benchmarks/moe-ep-20261021/baseline/scripts/kld.sh)
#   SPLIT as in bench.sh. BASE = cpu (the CPU logits of the baseline) or a .kld file made by "save" (self-consistency, SGLX 1 device vs N devices).
#   "save": write the logits of this run as the new base file $SG/kl/TAG.kld instead of comparing.
. "$(dirname "$0")/env.sh"
TAG=$1; SPLIT=$2; BASE=$3; MODE=$4
TS=${TS:-5/1}
mkdir -p "$SG/kl"
case "$SPLIT" in
  ""|-)  sglx_env ""; DEVS=Vulkan0,Vulkan1; EXTRA=(-ts $TS) ;;
  *)     sglx_env "$SPLIT"; DEVS=Vulkan0,Vulkan1,SGLX0; EXTRA=(-ts $TS/0 -ot 'token_embd\.weight=SGLX0') ;;
esac
[ -n "$FORCE_MMVQ" ] && export GGML_VK_FORCE_MMVQ=1
[ "$BASE" = cpu ] && BASE=$CPU_BASE
COMMON=(-f "$TEXT" -c 512 --chunks 4 -b ${KB:-128} -ub ${KUB:-128} -t 2 -fa off -dev "$DEVS" -ngl 99 -sm layer "${EXTRA[@]}")
if [ "$MODE" = save ]; then
  exec "$(dirname "$0")/run.sh" "$TAG" "$LB/llama-perplexity" -m "$MODEL" "${COMMON[@]}" --kl-divergence-base "$SG/kl/$TAG.kld"
fi
exec "$(dirname "$0")/run.sh" "$TAG" "$LB/llama-perplexity" -m "$MODEL" "${COMMON[@]}" --kl-divergence-base "$BASE" --kl-divergence
