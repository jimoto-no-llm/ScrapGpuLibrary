#!/bin/bash
# usage: bench.sh TAG CFG [llama-bench args...]   CFG per env.sh cfg(). Env: MODEL, NCMOE (default 0), UB (512), REPS (3), TS (e.g. 5/1), OT (-ot pattern), NGL (99)
. "$(dirname "$0")/env.sh"
TAG=$1; CFGN=$2; shift 2
cfg "$CFGN" || exit 2
EXTRA=()
[ -n "$TS" ] && EXTRA+=(-ts "$TS")
[ -n "$OT" ] && EXTRA+=(-ot "$OT")
[ -n "$MMAP" ] && EXTRA+=(-mmp "$MMAP")
ARGS=("$@")
[ ${#ARGS[@]} -eq 0 ] && ARGS=(-p 512 -n 128 -pg 512,128)
NG=${NGL:-99}; [ "$DEV" = none ] && NG=0
exec "$(dirname "$0")/run.sh" "$TAG" "$LB/llama-bench" -m "$MODEL" -dev "$DEV" -ngl $NG -fa 0 -t 2 -b "${UB:-512}" -ub "${UB:-512}" -r "${REPS:-3}" -sm layer -ncmoe "${NCMOE:-0}" "${EXTRA[@]}" -o jsonl "${ARGS[@]}"
