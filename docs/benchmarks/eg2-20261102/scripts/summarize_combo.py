#!/usr/bin/env python3
"""Table of the combination runs: logs/c_<name>.txt.  usage: summarize_combo.py LOGDIR [glob-substring]"""
import re, sys, glob, os
d = sys.argv[1]; sub = sys.argv[2] if len(sys.argv) > 2 else ""
print("| run | members (embeddings/s each) | SUM | start spread (s) |")
print("|---|---|---:|---:|")
for f in sorted(glob.glob(os.path.join(d, "c_*.txt"))):
    n = os.path.basename(f)[2:-4]
    if sub not in n: continue
    t = open(f).read()
    mem = re.findall(r"^(\S+) (\S+) WINDOW L=(\d+) B=(\d+) .*?emb_per_s=([\d.]+)", t, re.M)
    s = re.search(r"SUM emb_per_s=([\d.]+)\s+start_spread_s=([\d.]+)", t)
    if not s: continue
    print(f"| {n} | " + ", ".join(f"{c} {ty} B{b}: {float(e):.2f}" for c, ty, l, b, e in mem) + f" | **{float(s.group(1)):.2f}** | {s.group(2)} |")
