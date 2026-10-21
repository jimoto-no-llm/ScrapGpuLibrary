#!/usr/bin/env python3
"""summarize.py CSV...: per sampler CSV, the share of samples per value for each column while the GPU is busy (rx_busy>0 or wx_busy>0)."""
import csv, sys, collections
for f in sys.argv[1:]:
    rows = list(csv.DictReader(open(f)))
    busy = [r for r in rows if (r['rx_busy'] not in ('', '0') or r['wx_busy'] not in ('', '0'))]
    print(f"== {f}: {len(rows)} samples ({rows[-1]['t']} s), busy {len(busy)}")
    for c in rows[0]:
        if c in ('t',) or c.endswith('_temp') or c.endswith('_pwr') or c.endswith('_freq1') and False: continue
        if c.endswith('_temp') : continue
        cnt = collections.Counter(r[c] for r in busy)
        if c.endswith('_pwr') or c.endswith('_freq1') or c.endswith('_khz'):
            v = [float(r[c]) for r in busy if r[c]]
            if v: print(f"  {c}: mean {sum(v)/len(v):.0f} min {min(v):.0f} max {max(v):.0f}")
            continue
        top = cnt.most_common(4)
        print(f"  {c}: " + "  ".join(f"{k or '-'}={100*v/len(busy):.0f}%" for k, v in top))
