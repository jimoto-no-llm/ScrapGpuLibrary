#!/bin/bash
cd "$(dirname "$0")"
# -ot variant: all non-expert tensors on RX (-ts 48/0); experts of layers N..N+4 on WX 2100, later layers stay on RX, earlier layers on CPU (-ncmoe)
for n in 34 33; do
  OT="blk\\.($n|$((n+1))|$((n+2))|$((n+3))|$((n+4)))\\.ffn_.*_exps\\.weight=Vulkan1" ./sweep.sh amd ot_wx5 "$n" 48/0
done
echo Q6DONE
