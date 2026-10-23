#!/usr/bin/python3
"""sampler.py OUT.json INTERVAL_S [--gpu] -- cmd...
Runs cmd; samples per-thread schedstat/state (and optionally gpu_busy_percent of RX/WX) until exit.
Records final rusage via wait4. Output: JSON with samples and rusage."""
import os, sys, time, json, subprocess, resource
out, iv = sys.argv[1], float(sys.argv[2])
rest = sys.argv[3:]
gpu = False
if rest[0] == '--gpu': gpu = True; rest = rest[1:]
assert rest[0] == '--'; cmd = rest[1:]
RX='/sys/class/drm/card1/device/gpu_busy_percent'; WX='/sys/class/drm/card5/device/gpu_busy_percent'
env = dict(os.environ)
if 'CHILD_LD_PRELOAD' in env: env['LD_PRELOAD'] = env.pop('CHILD_LD_PRELOAD')
p = subprocess.Popen(cmd, env=env)
pid = p.pid
samples = []; last = {}
def rd(path):
    try:
        with open(path) as f: return f.read()
    except Exception: return None
t0 = time.time()
fr = open(RX) if gpu else None; fw = open(WX) if gpu else None
def rdf(f):
    try:
        f.seek(0); return int(f.read())
    except Exception: return -1
while True:
    r = os.wait4(pid, os.WNOHANG)
    if r[0] == pid: ru = r[2]; break
    t = time.time() - t0
    s = {'t': round(t, 4)}
    if gpu:
        s['rx'] = rdf(fr); s['wx'] = rdf(fw)
        k = len(samples)
        if k % 5 == 0:
            for nm, d in (('rx', 'card1'), ('wx', 'card5')):
                for kind in ('vram_used', 'gtt_used'):
                    v = rd(f'/sys/class/drm/{d}/device/mem_info_{kind}')
                    s[f'{nm}_{kind}'] = int(v) if v else -1
                for kind in ('sclk', 'mclk'):
                    txt = rd(f'/sys/class/drm/{d}/device/pp_dpm_{kind}') or ''
                    idx = [l for l in txt.splitlines() if l.rstrip().endswith('*')]
                    s[f'{nm}_{kind}'] = idx[0].strip() if idx else ''
    else:
        th = {}
        try:
            for tid in os.listdir(f'/proc/{pid}/task'):
                ss = rd(f'/proc/{pid}/task/{tid}/schedstat'); st = rd(f'/proc/{pid}/task/{tid}/stat')
                if ss is None or st is None: continue
                r = st.rsplit(')', 1)
                name = r[0].split('(', 1)[1]; f = r[1].split()
                th[tid] = [name, f[0], int(f[11]), int(f[12]), int(ss.split()[0])]  # state,utime,stime,run_ns
        except Exception: pass
        s['th'] = th
        for k, v in th.items(): last[k] = v
    samples.append(s)
    time.sleep(iv)
json.dump({'cmd': cmd, 'iv': iv, 'samples': samples, 'last': last, 'utime': ru.ru_utime, 'stime': ru.ru_stime,
           'wall': time.time() - t0, 'minflt': ru.ru_minflt, 'nvcsw': ru.ru_nvcsw, 'nivcsw': ru.ru_nivcsw}, open(out, 'w'))
