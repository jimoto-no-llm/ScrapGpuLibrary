#!/bin/bash
# usage: bench_card.sh CARD TYPE "L:B[:REPS] ..." [extra eg2bench args]
# One flock'd run per (card, type, L, B). Output: $LOGS/b_<CARD>_<TYPE>_L<L>_B<B>[_<TAG>].out (a RESULT line)
. "$(dirname "$0")/env.sh"; . "$(dirname "$0")/card_env.sh"
CARD=$1; TYPE=$2; CONFS=$3; shift 3
card_env $CARD || exit 1
for c in $CONFS; do
  IFS=: read L B R <<< "$c"; R=${R:-10}
  $RUN b_${CARD}_${TYPE}_L${L}_B${B}${TAG:+_$TAG} $EG2 -m $MODELS/eg2-$TYPE.gguf -ngl ${NGL:-99} --dev $DEVNAME -t ${THREADS:-2} --warm 2 --bench $L $B $R "$@"
done
