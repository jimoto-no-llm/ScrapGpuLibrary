# CPU affinity: the RX 6400 process on core 0, the others on core 1 (Celeron G3930: 2 cores, no SMT), against the unpinned runs
S=$HOME/moe-ep/eg2/scripts
export PINMAP="rx6400-vk=0 default=1"
bash $S/combo.sh rx6400-vk+wx2100-vk+gt730+gt710+gt430_pin_L128 30 "rx6400-vk:Q8_0:128:8 wx2100-vk:Q8_0:128:1 gt730:Q8_0:128:1 gt710:Q8_0:128:1 gt430:Q8_0:128:1"
bash $S/combo.sh rx6400-vk+wx2100-vk+gt730_pin_L128 30 "rx6400-vk:Q8_0:128:8 wx2100-vk:Q8_0:128:1 gt730:Q8_0:128:1"
bash $S/combo.sh rx6400-vk+wx2100-vk+gt730+gt710+gt430_pin_L32 30 "rx6400-vk:Q8_0:32:16 wx2100-vk:Q8_0:32:16 gt730:Q8_0:32:16 gt710:Q8_0:32:16 gt430:Q8_0:32:16"
bash $S/combo.sh rx6400-vk+wx2100-vk+gt730_pin_L32 30 "rx6400-vk:Q8_0:32:16 wx2100-vk:Q8_0:32:16 gt730:Q8_0:32:16"
echo QUEUE_DONE
