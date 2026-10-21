#!/bin/bash
# usage: bench.sh TAG BACKEND NCMOE REPS "P-LIST" "N-LIST" "PG-LIST" [UB] [EXTRA llama-bench args...]
# One llama-bench run of one single-device backend (see env.sh use_env). NCMOE=all keeps every expert on the GPU (--n-cpu-moe 0),
# NCMOE=cpu uses -dev none -ngl 0.
. "$(dirname "$0")/env.sh"
TAG=$1; BK=$2; NC=$3; R=$4; PL=$5; NL=$6; PG=$7; UB=${8:-128}; shift 8 2>/dev/null; EXTRA=("$@")
use_env "$BK" || exit 2
ARGS=(-m "$MODEL" -dev "$DEV" -ngl 99 -fa 0 -t 2 -b "$UB" -ub "$UB" -r "$R" -o jsonl)
[ "$BK" = cpu ] && ARGS=(-m "$MODEL" -dev none -ngl 0 -fa 0 -t 2 -b "$UB" -ub "$UB" -r "$R" -o jsonl)
[ "$BK" != cpu ] && ARGS+=(-ncmoe "$NC")
[ -n "$PL" ] && ARGS+=(-p "$PL")
[ -n "$NL" ] && ARGS+=(-n "$NL")
[ -n "$PG" ] && ARGS+=(-pg "$PG")
exec "$(dirname "$0")/run.sh" "$TAG" "$LB/llama-bench" "${ARGS[@]}" "${EXTRA[@]}"
