#!/usr/bin/env python3
"""Retrieval quality from one eg2bench dump of texts_retrieval.txt (N queries then N docs): recall@1, recall@5, MRR.
usage: retrieval.py dump.bin"""
import struct, sys, math
b = open(sys.argv[1], "rb").read()
n2, d = struct.unpack("ii", b[:8]); f = struct.unpack(f"{n2*d}f", b[8:8 + 4 * n2 * d])
rows = []
for i in range(n2):
    r = f[i * d:(i + 1) * d]; s = math.sqrt(sum(x * x for x in r)) or 1.0; rows.append([x / s for x in r])
n = n2 // 2; Q, D = rows[:n], rows[n:]
r1 = r5 = 0; mrr = 0.0
for i, q in enumerate(Q):
    sc = [sum(x * y for x, y in zip(q, dd)) for dd in D]
    rank = 1 + sum(1 for s in sc if s > sc[i])
    r1 += rank == 1; r5 += rank <= 5; mrr += 1.0 / rank
print(f"n={n} recall@1={r1/n:.4f} recall@5={r5/n:.4f} mrr={mrr/n:.4f}")
