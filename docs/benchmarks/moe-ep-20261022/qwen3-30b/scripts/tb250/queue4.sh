#!/bin/bash
cd "$(dirname "$0")"
./sweep.sh mix4 mix4_ts37-5-2-4 "31 30 29 28" 37/5/2/4
./sweep.sh mix4 mix4_ts38-5-2-3 "31 30 29" 38/5/2/3
./sweep.sh mix4 mix4_ts36-5-2-5 "30 29 28" 36/5/2/5
echo Q4DONE
