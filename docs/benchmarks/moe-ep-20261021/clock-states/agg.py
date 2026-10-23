#!/usr/bin/env python3
"""agg.py LABEL=glob ...: pool the sampler CSVs of one state and print the distributions that matter (RX sclk / busy / hwmon freq, WX, CPU MHz)."""
import csv, sys, glob, collections, statistics as st
for arg in sys.argv[1:]:
    lab, pat = arg.split('=', 1); rows = []
    for f in sorted(glob.glob(pat)): rows += list(csv.DictReader(open(f)))
    print(f"== {lab}: {len(rows)} samples from {len(glob.glob(pat))} files")
    for k in ('rx', 'wx'):
        fr = [float(r[f'{k}_freq1']) / 1e6 for r in rows if r[f'{k}_freq1']]
        bz = [float(r[f'{k}_busy']) for r in rows if r[f'{k}_busy']]
        hi = [f for f, b in zip(fr, bz) if b >= 50]
        print(f"  {k}: freq1 MHz mean {st.mean(fr):.0f} | samples busy>=50%: {len(hi)} ({100*len(hi)/len(rows):.0f}%), their freq mean {st.mean(hi) if hi else 0:.0f} min {min(hi) if hi else 0:.0f} | mean busy {st.mean(bz):.1f}")
    for c in ('rx_sclk', 'rx_fclk', 'wx_sclk', 'rx_pwr'):
        if c.endswith('pwr'):
            v = [float(r[c]) for r in rows if r[c]]; print(f"  {c}: mean {st.mean(v)/1e6:.1f} W max {max(v)/1e6:.1f}")
        else:
            cnt = collections.Counter(r[c] for r in rows); print(f"  {c}: " + "  ".join(f"{k}={100*v/len(rows):.0f}%" for k, v in cnt.most_common(5)))
    cp = [(float(r['cpu0_khz']) + float(r['cpu1_khz'])) / 2000 for r in rows]
    print(f"  cpu MHz mean {st.mean(cp):.0f}")
