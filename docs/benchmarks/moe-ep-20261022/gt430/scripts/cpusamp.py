#!/usr/bin/env python3
# cpusamp.py SECONDS: every 0.5 s the CPU time (user+sys, in % of one core) of the processes whose name starts with llama-bench / scrapgpu-device,
# and of each thread of llama-bench; the mean over the samples where llama-bench ran at more than 50 % is printed at the end
import os, sys, time, signal
dur = float(sys.argv[1])
def procs():
    out = {}
    for p in os.listdir('/proc'):
        if not p.isdigit(): continue
        try:
            comm = open('/proc/%s/comm' % p).read().strip()
        except Exception: continue
        if comm.startswith('llama-bench') or comm.startswith('scrapgpu-device'):
            out[int(p)] = comm
    return out
def ticks(pid):
    try:
        f = open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()
        return int(f[11]) + int(f[12])
    except Exception: return None
def thr(pid):
    r = {}
    try:
        for t in os.listdir('/proc/%d/task' % pid):
            try:
                f = open('/proc/%d/task/%s/stat' % (pid, t)).read()
                name = f.split('(', 1)[1].rsplit(')', 1)[0]
                v = f.rsplit(')', 1)[1].split()
                r[int(t)] = (name, int(v[11]) + int(v[12]))
            except Exception: pass
    except Exception: pass
    return r
hz = os.sysconf('SC_CLK_TCK')
t_end = time.time() + dur
prev = {}; prev_t = time.time(); rows = []
tprev = {}
stop = [False]
signal.signal(signal.SIGTERM, lambda *a: stop.__setitem__(0, True))
while time.time() < t_end and not stop[0]:
    time.sleep(0.5)
    now = time.time(); dt = now - prev_t; prev_t = now
    cur = {}
    for pid, comm in procs().items():
        k = ticks(pid)
        if k is not None: cur[pid] = (comm, k)
    row = {}
    for pid, (comm, k) in cur.items():
        if pid in prev:
            row[comm] = row.get(comm, 0.0) + 100.0 * (k - prev[pid][1]) / hz / dt
    prev = cur
    for pid, (comm, k) in cur.items():
        if comm.startswith('llama-bench'):
            for t, (n, v) in thr(pid).items():
                if t in tprev:
                    row['T:' + n] = row.get('T:' + n, 0.0) + 100.0 * (v - tprev[t]) / hz / dt
                tprev[t] = v
    if row.get('llama-bench', 0) > 50:
        rows.append(row)
keys = sorted({k for r in rows for k in r})
print('samples with llama-bench > 50%%: %d' % len(rows))
for k in keys:
    vals = [r.get(k, 0.0) for r in rows]
    if vals and sum(vals) / len(vals) > 1.0:
        print('%-28s mean %6.1f %%   max %6.1f %%' % (k, sum(vals) / len(vals), max(vals)))
