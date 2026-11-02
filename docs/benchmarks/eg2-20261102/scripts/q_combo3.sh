# L=128 with the RX 6400 at its best batch size (B=4, from the extra points); the others B=1
S=$HOME/moe-ep/eg2/scripts
R="rx6400-vk:Q8_0:128:4"
bash $S/combo.sh rx6400-vk_b4_L128 30 "$R"
bash $S/combo.sh rx6400-vk+wx2100-vk_b4_L128 30 "$R wx2100-vk:Q8_0:128:1"
bash $S/combo.sh rx6400-vk+wx2100-vk+gt730_b4_L128 30 "$R wx2100-vk:Q8_0:128:1 gt730:Q8_0:128:1"
bash $S/combo.sh rx6400-vk+wx2100-vk+gt730+gt710+gt430_b4_L128 30 "$R wx2100-vk:Q8_0:128:1 gt730:Q8_0:128:1 gt710:Q8_0:128:1 gt430:Q8_0:128:1"
bash $S/combo.sh rx6400-vk_b4_L128_r2 30 "$R"
bash $S/combo.sh rx6400-vk+wx2100-vk+gt730+gt710+gt430_b4_L128_r2 30 "$R wx2100-vk:Q8_0:128:1 gt730:Q8_0:128:1 gt710:Q8_0:128:1 gt430:Q8_0:128:1"
echo QUEUE_DONE
