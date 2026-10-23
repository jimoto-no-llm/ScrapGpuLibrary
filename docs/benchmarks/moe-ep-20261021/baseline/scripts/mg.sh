#!/bin/bash
# usage: mg.sh TAG CONFIG TS REPS "P-LIST" "N-LIST" "PG-LIST" [UB] [NCMOE]   multi-GPU llama-bench of one configuration (env.sh use_mg)
. "$(dirname "$0")/env.sh"
TAG=$1; CF=$2; TS=$3; R=$4; PL=$5; NL=$6; PG=$7; UB=${8:-128}; NC=${9:-0}
use_mg "$CF" || exit 2
ARGS=(-m "$MODEL" -dev "$DEV" -ngl 99 -fa 0 -t 2 -b "$UB" -ub "$UB" -r "$R" -ts "${TS//,//}" -sm layer -ncmoe "$NC" -o jsonl)
[ -n "$PL" ] && ARGS+=(-p "$PL")
[ -n "$NL" ] && ARGS+=(-n "$NL")
[ -n "$PG" ] && ARGS+=(-pg "$PG")
exec "$(dirname "$0")/run.sh" "$TAG" "$LB/llama-bench" "${ARGS[@]}"
