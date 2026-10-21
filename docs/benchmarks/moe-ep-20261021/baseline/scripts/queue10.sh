#!/bin/bash
S="$(dirname "$0")"
"$S/final_mg.sh" amdvk 9,2 512
"$S/final_mg.sh" amdvk 5,1 1024
"$S/final_mg.sh" amdvk 9,2 1024
echo QUEUE10_DONE
