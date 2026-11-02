# full matrix: every path x {Q8_0, Q4_0} x L/B configs, with a CPU reference run (Q8_0, L=128, B=1) between the cards (alternating measurement)
S=$HOME/moe-ep/eg2/scripts
CONFS="32:1:30 32:16:5 128:1:10 128:8:3 512:1:5 512:2:3 512:4:3"
for c in rx6400-vk wx2100-vk gt730 gt710 gt430 rx6400-cl wx2100-cl wx2100-clover; do
  TAG=cpuref THREADS=2 NGL=0 bash $S/bench_card.sh cpu Q8_0 "128:1:5"
  for t in Q8_0 Q4_0; do TAG=mx bash $S/bench_card.sh $c $t "$CONFS"; done
done
TAG=cpuref THREADS=2 NGL=0 bash $S/bench_card.sh cpu Q8_0 "128:1:5"
for t in Q8_0 Q4_0; do TAG=mx NGL=0 bash $S/bench_card.sh cpu $t "32:1:5 128:1:3 512:1:2"; done
echo QUEUE_DONE
