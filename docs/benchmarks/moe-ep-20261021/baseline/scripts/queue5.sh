#!/bin/bash
S="$(dirname "$0")"
"$S/final_single.sh" rx-vk 5 512
"$S/final_single.sh" rx-vk 6 512
"$S/final_single.sh" rx-vk 0 512
"$S/final_single.sh" wx-vk 17 128
"$S/final_single.sh" gt730 22 128
"$S/final_single.sh" gt430 24 128
"$S/final_single.sh" gt710 20 128
"$S/final_single.sh" rx-cl11 14 128
"$S/final_single.sh" rx-cl11 6 128
"$S/final_single.sh" wx-cl11 18 128
"$S/final_single.sh" cpu 0 128
echo QUEUE5_DONE
