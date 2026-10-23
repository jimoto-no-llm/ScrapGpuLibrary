#!/usr/bin/env python3
# mm_an.py FILE.cli [skip]: per MUL_MAT (N=1) the time of every event after mm_b, from the composite / pdev trace
import sys, collections
f = sys.argv[1]; skip = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ev = []
for l in open(f):
    t, tid, tag, a = l.split(); ev.append((int(t), tag, int(a)))
groups = []; cur = None
for t, tag, a in ev:
    if tag == 'mm_b': cur = {'t0': t, 'e': collections.OrderedDict()}; groups.append(cur)
    elif cur is not None:
        if tag in ('c_send_b','c_send_e','c_resp_hdr'): continue
        k = (tag, a if tag in ('sub_b','sub_e','wait_e') else 0)
        if k not in cur['e']: cur['e'][k] = (t - cur['t0']) / 1000.0
groups = groups[skip:]
keys = []
for g in groups:
    for k in g['e']:
        if k not in keys: keys.append(k)
print('MUL_MAT N=1 calls', len(groups))
print('%-16s %6s %8s %8s %8s %8s' % ('event', 'n', 'mean', 'p10', 'p50', 'p90'))
for k in keys:
    v = sorted(g['e'][k] for g in groups if k in g['e'])
    if len(v) < len(groups) // 2: continue
    print('%-16s %6d %8.0f %8.0f %8.0f %8.0f' % (k[0] + ('/%d' % k[1] if k[0] in ('sub_b','sub_e','wait_e') else ''), len(v), sum(v) / len(v), v[len(v)//10], v[len(v)//2], v[len(v)*9//10]))
