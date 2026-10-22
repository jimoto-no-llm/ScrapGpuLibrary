#!/usr/bin/env python3
"""Cross-corpus check of a static hot set: choose the top-K experts of each layer on corpus A, report the share of corpus B's routed
(token, slot) pairs that land inside it (coverage). Also reports how many experts per layer cover 50 / 80 % of the traffic.
usage: python3 -I compare.py a.counts.json b.counts.json ... (files written by expert_freq.py --json)"""
import json, sys, itertools
data = {}
for p in sys.argv[1:]:
    data[p.split('/')[-1].split('.')[0]] = {int(k): v for k, v in json.load(open(p)).items()}
def hot(c, k): return set(sorted(range(len(c)), key=lambda i: -c[i])[:k])
def cover(a, b, k):
    num = den = 0
    for l in a:
        h = hot(a[l], k); num += sum(b[l][i] for i in h); den += sum(b[l])
    return num / den
print('coverage of corpus B traffic by the top-K set chosen on corpus A (mean over 48 layers; uniform = K/128)')
print('A \\ B       ' + '  '.join('%-9s' % n for n in data) + '   K')
for K in (16, 32, 48, 64):
    for a in data:
        print('%-12s' % a + '  '.join('%-9.3f' % cover(data[a], data[b], K) for b in data) + '   %d (uniform %.3f)' % (K, K / 128))
print()
for n, d in data.items():
    n50 = []; n80 = []
    for l in d:
        c = sorted(d[l], reverse=True); t = sum(c); s = 0
        for i, x in enumerate(c):
            s += x
            if s >= 0.5 * t and not n50 or (s >= 0.5 * t and len(n50) < len(n80) + 1 and False): pass
        # simple count
        def k_for(f):
            s = 0
            for i, x in enumerate(c):
                s += x
                if s >= f * t: return i + 1
        n50.append(k_for(0.5)); n80.append(k_for(0.8))
    print('%-9s experts needed per layer (of 128): 50%% of traffic mean %.1f (min %d max %d), 80%% mean %.1f (min %d max %d)' % (
        n, sum(n50) / len(n50), min(n50), max(n50), sum(n80) / len(n80), min(n80), max(n80)))
