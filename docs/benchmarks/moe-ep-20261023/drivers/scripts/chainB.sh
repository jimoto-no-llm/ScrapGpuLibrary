#!/bin/bash
cd ~/moe-ep/drivers/scripts
./probe_state.sh b_nv470
./bench_cuda470.sh
./bench_cl11_cards.sh NVIDIA gt730 gt710
./bench_cl11_470.sh
echo CHAIN_DONE
