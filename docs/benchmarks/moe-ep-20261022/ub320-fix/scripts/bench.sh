#!/bin/bash
# usage: bench.sh TAG SPLIT [llama-bench args...]
#   SPLIT = GGML_SGLX_SPLIT ("" or "-": the plain C1 without the plugin; "noop": C1 with a harmless -ot override, the control for the -ot runs)
# Environment: FORCE_MMVQ=1 sets GGML_VK_FORCE_MMVQ=1; TS (default 5/1) is the layer split of the two AMD cards; PROFILE=0/1 (default 1).
# Default measurement: llama-bench -p 512 -n 128 -pg 512,128 -r 5 -ub 512 (the baseline's C1 command with the output projection on SGLX0).
. "$(dirname "$0")/env.sh"
TAG=$1; SPLIT=$2; shift 2
TS=${TS:-5/1}
case "$SPLIT" in
  ""|-)  sglx_env ""; DEVS=Vulkan0/Vulkan1; EXTRA=(-ts $TS) ;;
  sglxnoop) sglx_env "Vulkan0:0.5,Vulkan1:0.2,CL11_0:0.2,CL11_1:0.1"; DEVS=Vulkan0/Vulkan1/SGLX0; EXTRA=(-ts $TS/0 -ot "no_such_tensor_zz=CPU") ;;
  noop)  sglx_env ""; DEVS=Vulkan0/Vulkan1; EXTRA=(-ts $TS -ot "no_such_tensor_zz=CPU") ;;
  *)     sglx_env "$SPLIT"; DEVS=Vulkan0/Vulkan1/SGLX0
         if [ -n "$NOOT" ]; then EXTRA=(-ts ${TS3:-0.82/0.12/0.06}); else EXTRA=(-ts $TS/0 -ot "${OTX:+$OTX;}token_embd\.weight=SGLX0"); fi ;;
esac
[ -n "$FORCE_MMVQ" ] && export GGML_VK_FORCE_MMVQ=1
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(-p 512 -n 128 -pg 512,128)
exec "$(dirname "$0")/run.sh" "$TAG" "$LB/llama-bench" -m "$MODEL" -dev "$DEVS" -ngl 99 -fa ${FA:-0} -t 2 -b ${B:-512} -ub "${UB:-512}" -r "${REPS:-5}" -sm layer -ncmoe 0 "${EXTRA[@]}" -o jsonl "${ARGS[@]}"
