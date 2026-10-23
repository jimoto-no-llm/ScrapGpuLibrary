#!/usr/bin/env python3
"""summarize.py LOGDIR [prefix...]  -> one line per run: tag, generated tokens, tg t/s, drafted, accepted, accept %, mean tokens/step, rc.
llama-completion logs: 'eval time' line (decode only).  llama-speculative-simple logs: 'decoded N tokens in S seconds' (decode loop incl. draft)."""
import re, sys, glob, os
d = sys.argv[1]; pre = sys.argv[2:]
rows = []
for f in sorted(glob.glob(os.path.join(d, "*.log"))):
    tag = os.path.basename(f)[:-4]
    if pre and not any(tag.startswith(p) for p in pre): continue
    t = open(f, errors="replace").read()
    rc = re.search(r"# rc=(\d+)", t)
    rc = rc.group(1) if rc else "?"
    m = re.search(r"decoded\s+(\d+) tokens in\s+([\d.]+) seconds, speed:\s+([\d.]+) t/s", t)
    ds = re.search(r"n_drafted = (\d+)", t); ac = re.search(r"n_accept\s+= (\d+)", t)
    if m:
        n = int(m.group(1)); s = float(m.group(2)); tps = float(m.group(3))
        dr = int(ds.group(1)) if ds else 0; a = int(ac.group(1)) if ac else 0
        steps = n - a
        rows.append((tag, n, tps, dr, a, 100.0*a/dr if dr else 0, n/steps if steps else 0, rc))
        continue
    m = re.search(r"eval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second", t)
    if m and "prompt eval" in t:
        rows.append((tag, int(m.group(2)), float(m.group(3)), 0, 0, 0, 1.0, rc)); continue
    rows.append((tag, 0, 0, 0, 0, 0, 0, rc))
print("| tag | tokens | tg t/s | drafted | accepted | accept % | tokens/step | rc |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for r in rows:
    print("| %s | %d | %.2f | %d | %d | %.1f | %.2f | %s |" % r)
