#!/bin/bash
# usage: corr_card.sh CARD "TYPES"   dump the correctness set on the card, compare with the CPU of the same type (gate: mean cos >= 0.999)
# and with the CPU BF16 reference (informational). Appends to $LOGS/corr.tsv
. "$(dirname "$0")/env.sh"; . "$(dirname "$0")/card_env.sh"
CARD=$1; TYPES=$2; card_env $CARD || exit 1
for t in $TYPES; do
  D=$DUMPS/${CARD}${TAG:+_$TAG}_${t}_corr.bin
  TMO=1800 $RUN corr_${CARD}${TAG:+_$TAG}_$t $EG2 -m $MODELS/eg2-$t.gguf -ngl 99 --dev $DEVNAME -t 2 --max-tokens 1500 --dump $TASK/texts_corr.txt $D
  if [ -s $D ]; then
    same=$(python3 $SG/cmp_emb.py $DUMPS/cpu_${t}_corr.bin $D); bf=$(python3 $SG/cmp_emb.py $DUMPS/cpu_bf16_corr.bin $D)
  else same="NO DUMP"; bf="NO DUMP"; fi
  printf "%s\t%s\t%s\tvs CPU %s: %s\tvs CPU BF16: %s\n" "$(date +%F_%T)" "$CARD${TAG:+_$TAG}" "$t" "$t" "$same" "$bf" | tee -a $LOGS/corr.tsv
done
