# extra points: larger batches of short texts, small batches of 128-token texts, and layer splits over two cards in one process
S=$HOME/moe-ep/eg2/scripts
for c in rx6400-vk wx2100-vk rx6400-cl wx2100-cl gt730 gt710 gt430; do
  t=Q8_0; case $c in rx6400-vk|wx2100-vk) t=Q4_0;; esac
  TAG=ex bash $S/bench_card.sh $c $t "32:32:5 32:64:3 128:2:5 128:4:3"
done
TAG=ex bash $S/bench_card.sh vk-pair Q4_0 "128:1:5 128:8:3 32:16:5" --ts 3,1
TAG=ex bash $S/bench_card.sh kepler-pair Q8_0 "128:1:5 128:8:3 32:16:5" --ts 2,1
echo QUEUE_DONE
