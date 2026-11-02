#!/usr/bin/env python3
"""Per path and type: embeddings/s at B=1 and the best over every measured batch size, for L = 32 / 128 / 512 (tags mx and ex).  usage: best_table.py LOGDIR"""
import re, sys, glob, os, collections
d = sys.argv[1]
R = collections.defaultdict(dict)
for f in glob.glob(os.path.join(d, "b_*_*.out")):
    m = re.match(r"b_(.+?)_(Q8_0|Q4_0)_L(\d+)_B(\d+)_(mx|ex)\.out", os.path.basename(f))
    if not m: continue
    card, typ, L, B = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    r = re.search(r"emb_per_s_mean=([\d.]+)", open(f).read())
    if r: R[(card, typ)][(L, B)] = float(r.group(1))
print("| path | type | L32 B=1 | L32 best | L128 B=1 | L128 best | L512 B=1 | L512 best |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for (card, typ), v in sorted(R.items()):
    cells = []
    for L in (32, 128, 512):
        b1 = v.get((L, 1)); bs = [(e, B) for (l, B), e in v.items() if l == L]
        best = max(bs) if bs else None
        cells += [f"{b1:.2f}" if b1 else "-", f"**{best[0]:.2f}** (B={best[1]})" if best else "-"]
    print(f"| {card} | {typ} | " + " | ".join(cells) + " |")
