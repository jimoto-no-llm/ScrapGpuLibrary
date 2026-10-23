#!/bin/bash
# usage: kld.sh base                         CPU reference logits (-dev none -ngl 0)
#        kld.sh TAG single BACKEND NCMOE     one single-device configuration vs the reference
#        kld.sh TAG mg CONFIG TS             one multi-GPU configuration vs the reference
# llama-perplexity -c 512 --chunks 4 -b 128 -ub 128 -t 2 -fa off, the text $TEXT (README.md + docs/*.md of the checkout)
. "$(dirname "$0")/env.sh"
BASE=$ROOT/kl/base.kld
COMMON=(-f "$TEXT" -c 512 --chunks 4 -b 128 -ub 128 -t 2 -fa off)
if [ "$1" = base ]; then
  use_env cpu
  exec "$(dirname "$0")/run.sh" kld_base_cpu "$LB/llama-perplexity" -m "$MODEL" "${COMMON[@]}" -dev none -ngl 0 --kl-divergence-base "$BASE"
fi
TAG=$1; KIND=$2; shift 2
if [ "$KIND" = single ]; then
  use_env "$1" || exit 2; NC=$2
  exec "$(dirname "$0")/run.sh" "$TAG" "$LB/llama-perplexity" -m "$MODEL" "${COMMON[@]}" -dev "$DEV" -ngl 99 -ncmoe "$NC" --kl-divergence-base "$BASE" --kl-divergence
else
  use_mg "$1" || exit 2
  exec "$(dirname "$0")/run.sh" "$TAG" "$LB/llama-perplexity" -m "$MODEL" "${COMMON[@]}" -dev "${DEV//\//,}" -ngl 99 -ts "$2" -sm layer --kl-divergence-base "$BASE" --kl-divergence
fi
