# data-parallel combinations with the best batch size of each card for the length (Q8_0 everywhere, so the embeddings are comparable)
S=$HOME/moe-ep/eg2/scripts
bestB() { # card L
  case $2 in
    32) echo 16 ;;
    128) case $1 in rx6400-vk) echo 8;; *) echo 1;; esac ;;
    *) echo 1 ;;
  esac
}
mk() { local L=$1; shift; local out=""; for c in "$@"; do out="$out $c:Q8_0:$L:$(bestB $c $L)"; done; echo $out; }
for L in 128 32 512; do
  for c in "rx6400-vk" "rx6400-vk wx2100-vk" "rx6400-vk wx2100-vk gt730" "rx6400-vk wx2100-vk gt730 gt710 gt430" "wx2100-vk gt730 gt710 gt430"; do
    n=$(echo $c | tr ' ' '+')
    bash $S/combo.sh ${n}_bb_L$L 30 "$(mk $L $c)"
  done
done
for i in 2 3; do   # repeats of the L=128 extremes, to see the run-to-run spread
  bash $S/combo.sh rx6400-vk_bb_L128_r$i 30 "$(mk 128 rx6400-vk)"
  bash $S/combo.sh rx6400-vk+wx2100-vk+gt730+gt710+gt430_bb_L128_r$i 30 "$(mk 128 rx6400-vk wx2100-vk gt730 gt710 gt430)"
done
echo QUEUE_DONE
