#!/usr/bin/env python3
"""Markdown tables from logs/run/*.out (llama-bench jsonl). usage: python3 -I summarize.py logs/run"""
import json, glob, re, sys, os
d = sys.argv[1]
def load(f):
    r = {}
    for l in open(f):
        j = json.loads(l); r[(j['n_prompt'], j['n_gen'])] = (j['avg_ts'], j['stddev_ts'], len(j['samples_ts']))
    return r
def cmd(tag):
    for l in open(f'{d}/{tag}.log'):
        if l.startswith('# cmd:'): return l[7:].strip()
def perf(tag):
    for l in open(f'{d}/{tag}.log'):
        if l.startswith('# perf:'): return 'manual' if 'manual' in l else ('auto' if 'auto' in l else l)
print('## finals (r=3)\n\n| tag | perf | pp512 | tg128 | pg512,128 |\n| --- | --- | --- | --- | --- |')
for f in sorted(glob.glob(f'{d}/final*.out')):
    tag = os.path.basename(f)[:-4]; r = load(f)
    g = lambda k: ('%.2f ± %.2f' % r[k][:2]) if k in r else '-'
    print(f'| {tag} | {perf(tag)} | {g((512,0))} | {g((0,128))} | {g((512,128))} |')
print('\n## sweeps (pp128 / tg32)\n\n| tag | reps | pp128 | tg32 |\n| --- | --- | --- | --- |')
for f in sorted(glob.glob(f'{d}/sw_*.out'), key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', x)]):
    tag = os.path.basename(f)[:-4]; r = load(f)
    if (128,0) in r and (0,32) in r:
        print(f'| {tag} | {r[(0,32)][2]} | {r[(128,0)][0]:.2f} | {r[(0,32)][0]:.2f} |')
    else:
        print(f'| {tag} | - | FAILED | |')
