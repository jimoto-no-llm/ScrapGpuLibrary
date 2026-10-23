#!/bin/bash
S="$(dirname "$0")"
# C1: two AMD cards, official Vulkan, layer split
for ub in 128 256 512; do "$S/final_mg.sh" amdvk 5,1 $ub; done
# five cards, CL11 (docs/MULTI_GPU.md split and the best pp split of the exploration)
"$S/final_mg.sh" cl5 8,8,18,40,18 128
"$S/final_mg.sh" cl5 2,4,16,60,18 128
# CL11 (Kepler) + Vulkan (AMD), with and without GT 430 over RPC
"$S/final_mg.sh" mix4 1,4,60,20 128
"$S/final_mg.sh" mix5 2,2,8,50,20 128
"$S/final_mg.sh" cl4 4,12,50,24 128
echo QUEUE7_DONE
