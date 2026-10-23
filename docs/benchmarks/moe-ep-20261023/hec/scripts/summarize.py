#!/usr/bin/env python3
"""Tables from the hec_gen output of the runs (logs/*.out): random-token bench (pp512 / tg128 / pg512:128) and natural text.
usage: summarize.py LOGDIR [bench|text]"""
import json, os, re, sys, glob, statistics as st

def lines(path):
    out = []
    try:
        for l in open(path, errors='replace'):
            l = l.strip()
            if l.startswith('{'):
                try:
                    out.append(json.loads(l))
                except ValueError:
                    pass
    except OSError:
        pass
    return out

def perf(path):
    try:
        for l in open(path, errors='replace'):
            if l.startswith('# perf:'):
                return 'pin' if 'manual' in l else 'auto'
    except OSError:
        pass
    return '?'

def main():
    d = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'bench'
    rows = {}
    for p in sorted(glob.glob(os.path.join(d, '*.out'))):
        tag = os.path.basename(p)[:-4]
        js = lines(p)
        if not js:
            continue
        if mode == 'bench' and any('avg_ts' in j for j in js):
            rows[tag] = (perf(p[:-4] + '.log'), {j['test']: (j['avg_ts'], j['stddev_ts']) for j in js if 'avg_ts' in j and 'test' in j})
        if mode == 'text' and any(j.get('test', '').startswith('text:') for j in js):
            rows[tag] = (perf(p[:-4] + '.log'), {j['test']: j for j in js if j.get('test', '').startswith('text:')})
    if mode == 'bench':
        print('| tag | perf | pp512 | tg128 | pg512,128 |')
        print('| --- | --- | --- | --- | --- |')
        for tag, (pf, r) in rows.items():
            f = lambda k: ('%.2f ± %.2f' % r[k]) if k in r else '-'
            print('| %s | %s | %s | %s | %s |' % (tag, pf, f('pp512'), f('tg128'), f('pg512:128')))
    else:
        print('| tag | perf | prompt | n_prompt | pp t/s | tg t/s (decode) | tg t/s (with sampling) |')
        print('| --- | --- | --- | --- | --- | --- | --- |')
        for tag, (pf, r) in rows.items():
            for k, j in r.items():
                print('| %s | %s | %s | %d | %.1f | %.2f | %.2f |' % (tag, pf, k[5:], j['n_prompt'], j['pp_ts'], j['tg_ts'], j['tg_total_ts']))

main()
