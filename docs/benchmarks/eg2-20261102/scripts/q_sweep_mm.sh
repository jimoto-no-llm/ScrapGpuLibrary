# mat-mat kernel sweep (L=128, B=1, Q8_0): the Kepler candidates (GGML_CL11_MM_VARIANT) on GT 730 / GT 710, MM_BN on every CL11 path
S=$HOME/moe-ep/eg2/scripts
for c in gt730 gt710; do
  for v in "" x84 x84p x88 x88w x88s x88k x88d x164 x168 x168s x168w x88g; do
    GGML_CL11_MM_VARIANT=$v TAG=mm_${v:-def} bash $S/bench_card.sh $c Q8_0 "128:1:5"
  done
done
for c in gt730 gt710 gt430 rx6400-cl wx2100-cl wx2100-clover; do
  for bn in 4 8; do GGML_CL11_MM_BN=$bn TAG=mmbn$bn bash $S/bench_card.sh $c Q8_0 "128:1:5"; done
done
echo QUEUE_DONE
