# stage 1: which weight type is fastest on each path (L=128, B=1)
S=$HOME/moe-ep/eg2/scripts
for c in gt730 gt710 gt430 rx6400-vk wx2100-vk rx6400-cl wx2100-cl wx2100-clover; do
  for t in Q8_0 Q4_0 Q4_K_M F16; do TAG=base bash $S/bench_card.sh $c $t "128:1:5"; done
done
echo QUEUE_DONE
