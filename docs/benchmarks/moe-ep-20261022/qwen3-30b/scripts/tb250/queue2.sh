#!/bin/bash
cd "$(dirname "$0")"
for p in "43/5 37 36 35 34" "44/4 38 37 36" "42/6 36 35 34"; do
  set -- $p; ts=$1; shift
  ./sweep.sh amd amd_ts${ts/\//-} "$*" $ts
done
echo Q2DONE
