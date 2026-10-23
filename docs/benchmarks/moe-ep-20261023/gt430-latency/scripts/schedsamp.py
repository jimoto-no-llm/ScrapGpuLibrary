#!/usr/bin/env python3
# schedsamp.py SECONDS: every 0.5 s /proc/PID/task/TID/schedstat (run ns, run-queue wait ns, timeslices) and the CPU number of every thread of llama-bench and
# scrapgpu-device-server; at the end (or on SIGTERM) the delta between the first and the last sample of each thread, per second of the window
import os, sys, time, signal
dur = float(sys.argv[1])
stop = False
def h(*a):
    global stop; stop = True
signal.signal(signal.SIGTERM, h)
first = {}; last = {}; cpus = {}
t_first = t_last = None
end = time.time() + dur
while time.time() < end and not stop:
    now = time.time()
    for p in os.listdir('/proc'):
        if not p.isdigit(): continue
        try: comm = open('/proc/%s/comm' % p).read().strip()
        except Exception: continue
        if not (comm.startswith('llama-bench') or comm.startswith('scrapgpu-device')): continue
        try: tids = os.listdir('/proc/%s/task' % p)
        except Exception: continue
        for t in tids:
            try:
                name = open('/proc/%s/task/%s/comm' % (p, t)).read().strip()
                ss = open('/proc/%s/task/%s/schedstat' % (p, t)).read().split()
                st = open('/proc/%s/task/%s/stat' % (p, t)).read().rsplit(')', 1)[1].split()
                cpu = int(st[36])
            except Exception: continue
            k = (int(p), int(t))
            v = (name, int(ss[0]), int(ss[1]), int(ss[2]), now)
            if k not in first and now > 0 and (time.time() - (end - dur)) > 2: first[k] = v
            if k in first: last[k] = v
            cpus.setdefault(k, {}); cpus[k][cpu] = cpus[k].get(cpu, 0) + 1
    time.sleep(0.5)
print('%-8s %-8s %-16s %9s %9s %9s %s' % ('pid', 'tid', 'name', 'run ms/s', 'wait ms/s', 'slices/s', 'cpu samples'))
rows = []
for k, v0 in first.items():
    v1 = last[k]; dt = v1[4] - v0[4]
    if dt < 1: continue
    rows.append((v1[1] - v0[1], '%-8d %-8d %-16s %9.1f %9.1f %9.0f %s' % (k[0], k[1], v1[0], (v1[1] - v0[1]) / dt / 1e6, (v1[2] - v0[2]) / dt / 1e6, (v1[3] - v0[3]) / dt, dict(cpus[k]))))
for _, r in sorted(rows, reverse=True): print(r)
