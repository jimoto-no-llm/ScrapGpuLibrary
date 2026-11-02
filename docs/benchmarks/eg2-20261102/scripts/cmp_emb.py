#!/usr/bin/env python3
"""Compare two eg2bench dumps: cosine similarity per row (rows L2-normalised here), mean / min / p1, PASS if mean >= thr.
usage: cmp_emb.py ref.bin test.bin [--thr 0.999]   (pure Python, no numpy)"""
import struct, sys, math
def load(p):
    b = open(p, "rb").read()
    n, d = struct.unpack("ii", b[:8])
    f = struct.unpack(f"{n*d}f", b[8:8 + 4 * n * d])
    rows = []
    for i in range(n):
        r = f[i * d:(i + 1) * d]
        s = math.sqrt(sum(x * x for x in r)) or 1.0
        rows.append([x / s for x in r])
    return rows
a, b = load(sys.argv[1]), load(sys.argv[2])
thr = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.999
assert len(a) == len(b), (len(a), len(b))
c = sorted(sum(x * y for x, y in zip(r, s)) for r, s in zip(a, b))
mean = sum(c) / len(c)
bad = sum(1 for x in c if math.isnan(x))
print(f"n={len(c)} mean_cos={mean:.6f} min={c[0]:.6f} p1={c[max(0,len(c)//100)]:.6f} nan={bad} -> {'PASS' if mean >= thr and bad == 0 else 'FAIL'} (thr {thr})")
