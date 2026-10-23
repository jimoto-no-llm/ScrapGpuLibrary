#!/bin/bash
# coarse n-cpu-moe sweep (step 2, from "all experts on the CPU" downwards until a value does not run), -r 2, pp128 + tg64
. "$(dirname "$0")/env.sh"
V="24 22 20 18 16 14 12 10 8 6 4 2 0"
for bk in "$@"; do
  case "$bk" in
    *-vk) "$(dirname "$0")/sweep.sh" "$bk" 128 $V ;;
    *)    "$(dirname "$0")/sweep.sh" "$bk" 128 $V ;;
  esac
done
echo QUEUE_DONE
