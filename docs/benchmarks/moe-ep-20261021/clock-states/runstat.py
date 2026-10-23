#!/usr/bin/env python3
"""runstat.py TAG N [SPLIT]: N bare (unsampled) bench.sh runs of C1 (or SGLX SPLIT) back to back; per run tg128 plus the host-side cost of the run:
rusage of the children (user/sys/ctx switches/faults), /proc/stat and /proc/interrupts deltas, and /proc/pressure. One JSON line per run."""
import sys, os, json, time, resource, subprocess, re
sg = os.path.expanduser('~/moe-ep/sglx'); tag, n = sys.argv[1], int(sys.argv[2]); split = sys.argv[3] if len(sys.argv) > 3 else ''
def stat():
    out = {}
    for l in open('/proc/stat'):
        p = l.split()
        if p[0].startswith('cpu') or p[0] in ('ctxt', 'intr', 'softirq'): out[p[0]] = list(map(int, p[1:]))
    return out
def irq():
    out = {}
    for l in open('/proc/interrupts'):
        p = l.split()
        if len(p) < 3 or not p[0].endswith(':'): continue
        c = 0
        for x in p[1:]:
            if x.isdigit(): c += int(x)
            else: break
        out[p[0] + ' ' + ' '.join(p[-2:])] = c
    return out
def psi(): return open('/proc/pressure/cpu').read().split('\n')[0] if os.path.exists('/proc/pressure/cpu') else ''
for i in range(1, n + 1):
    t = f'{tag}_{i}'; s0, i0, r0, w0 = stat(), irq(), resource.getrusage(resource.RUSAGE_CHILDREN), time.time()
    subprocess.run([f'{sg}/bench.sh', f'cs_{t}', split], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ, PROFILE='0'))
    w1, r1, s1, i1 = time.time(), resource.getrusage(resource.RUSAGE_CHILDREN), stat(), irq()
    tg = None
    for l in open(f'{sg}/logs/cs_{t}.out'):
        try: d = json.loads(l)
        except Exception: continue
        if d['n_gen'] == 128 and d['n_prompt'] == 0: tg = round(d['avg_ts'], 2)
    cpu = {k: [b - a for a, b in zip(s0[k], s1[k])] for k in s0 if k.startswith('cpu')}
    irqd = sorted(((v - i0.get(k, 0), k) for k, v in i1.items() if v - i0.get(k, 0) > 2000), reverse=True)[:6]
    print(json.dumps(dict(t=t, clock=time.strftime('%H:%M:%S'), tg=tg, wall=round(w1 - w0, 1), user=round(r1.ru_utime - r0.ru_utime, 1), sys=round(r1.ru_stime - r0.ru_stime, 1),
        nvcsw=r1.ru_nvcsw - r0.ru_nvcsw, nivcsw=r1.ru_nivcsw - r0.ru_nivcsw, minflt=r1.ru_minflt - r0.ru_minflt, majflt=r1.ru_majflt - r0.ru_majflt,
        cpu_stat_user_nice_sys_idle_iowait_irq_softirq_steal=cpu, ctxt=s1['ctxt'][0] - s0['ctxt'][0], intr=s1['intr'][0] - s0['intr'][0], top_irq=irqd, psi=psi())), flush=True)
