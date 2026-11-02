#!/usr/bin/env python3
"""Tables from the RESULT lines of logs/b_<card>_<type>_L<L>_B<B>_<tag>.out.  usage: summarize.py LOGDIR TAG"""
import re, sys, glob, os, collections
d, tag = sys.argv[1], sys.argv[2]
rows = collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.join(d, f"b_*_{tag}.out"))):
    m = re.match(r"b_(.+?)_(Q8_0|Q4_0|Q4_K_M|F16|bf16)_L(\d+)_B(\d+)_" + re.escape(tag) + r"\.out", os.path.basename(f))
    if not m: continue
    card, typ, L, B = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    t = open(f).read()
    r = re.search(r"mean_ms=([\d.]+) sd_ms=([\d.]+).*emb_per_s_mean=([\d.]+)", t)
    if r:
        mean, sd = float(r.group(1)), float(r.group(2))
        rows[(card, typ)][(L, B)] = (float(r.group(3)), B * 1000.0 * sd / (mean * mean), mean)
    else:
        rows[(card, typ)][(L, B)] = None
cols = sorted({k for v in rows.values() for k in v})
print("| path | type | " + " | ".join(f"L{L} B{B}" for L, B in cols) + " |")
print("|---|---|" + "---:|" * len(cols))
for (card, typ), v in sorted(rows.items()):
    cells = []
    for c in cols:
        x = v.get(c)
        cells.append("-" if x is None else f"{x[0]:.2f} ± {x[1]:.2f}")
    print(f"| {card} | {typ} | " + " | ".join(cells) + " |")
print()
print("best over the measured batch sizes, per length (embeddings/s, batch size):")
print("| path | type | " + " | ".join(f"L{L}" for L in sorted({c[0] for c in cols})) + " |")
print("|---|---|" + "---:|" * len({c[0] for c in cols}))
for (card, typ), v in sorted(rows.items()):
    cells = []
    for L in sorted({c[0] for c in cols}):
        best = max(((x[0], B) for (l, B), x in v.items() if l == L and x), default=None)
        cells.append("-" if best is None else f"{best[0]:.2f} (B={best[1]})")
    print(f"| {card} | {typ} | " + " | ".join(cells) + " |")
