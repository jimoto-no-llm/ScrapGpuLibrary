# data-parallel combinations (whole model on every card, one process per card, common 30 s window)
S=$HOME/moe-ep/eg2/scripts
L=${L:-128}; B=${B:-1}; D=${DUR:-30}
declare -A SET
# "consistent": the same weights (Q8_0) on every card, so the embeddings are comparable;  "best": the fastest type of each card
m8() { local out=""; for c in "$@"; do out="$out $c:Q8_0:$L:$B"; done; echo $out; }
mb() { local out=""; for c in "$@"; do case $c in rx6400-vk|wx2100-vk) t=Q4_0;; gt430) t=Q4_0;; *) t=Q8_0;; esac; out="$out $c:$t:$L:$B"; done; echo $out; }
combos=(
 "rx6400-vk"
 "rx6400-vk wx2100-vk"
 "rx6400-vk gt730"
 "gt730 gt710"
 "gt730 gt710 gt430"
 "rx6400-vk wx2100-vk gt730"
 "rx6400-vk wx2100-vk gt730 gt710"
 "rx6400-vk wx2100-vk gt730 gt710 gt430"
)
i=0
for c in "${combos[@]}"; do
  i=$((i+1)); n=$(echo $c | tr ' ' '+')
  bash $S/combo.sh ${n}_q8_L${L}B${B} $D "$(m8 $c)"
  bash $S/combo.sh ${n}_best_L${L}B${B} $D "$(mb $c)"
done
echo QUEUE_DONE
