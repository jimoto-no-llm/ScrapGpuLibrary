#!/bin/bash
cd "$(dirname "$0")"
./sweep.sh mix4 mix4_ts38-5-2-3 "28 27 26" 38/5/2/3
./sweep.sh cl5b cl5b_ts35-5-4-2-2 "28 27 26" 35/5/4/2/2
./sweep.sh mix5b mix5b_ts36-5-4-2-1 "29 28 27" 36/5/4/2/1
echo Q5DONE
