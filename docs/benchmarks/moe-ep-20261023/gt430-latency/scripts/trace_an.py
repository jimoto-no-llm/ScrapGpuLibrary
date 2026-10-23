#!/usr/bin/env python3
# trace_an.py PREFIX [skip_jobs]: break the REMOTE0 job into segments from the client (.cli) and server (.srv) traces
import sys, statistics as st
pre = sys.argv[1]; skip = int(sys.argv[2]) if len(sys.argv) > 2 else 20
def load(f):
    out = []
    for l in open(f):
        t, tid, tag, a = l.split(); out.append((int(t), int(tid), tag, int(a)))
    return out
cli = load(pre + '.cli'); srv = load(pre + '.srv')
# client jobs
jobs = []; cur = None
for t, tid, tag, a in cli:
    if tag == 'p_submit': cur = {'submit': t}; jobs.append(cur)
    elif cur is not None:
        if tag == 'p_wake' and 'wake' not in cur: cur['wake'] = t
        elif tag == 'c_send_b' and a == 50 and 'wake' in cur and 'send_b' not in cur: cur['send_b'] = t
        elif tag == 'c_send_e' and a == 50 and 'send_b' in cur and 'send_e' not in cur: cur['send_e'] = t
        elif tag == 'c_resp_hdr' and a == 50 and 'send_e' in cur and 'resp' not in cur: cur['resp'] = t
        elif tag == 'p_exec_e' and 'exec_e' not in cur: cur['exec_e'] = t
        elif tag == 'p_wait_e' and 'exec_e' in cur and 'wait_e' not in cur: cur['wait_e'] = t
jobs = [j for j in jobs if 'exec_e' in j and 'resp' in j]
# server batches
bs = []; cur = None
for t, tid, tag, a in srv:
    if tag == 's_hdr': cur = {'hdr': t, 'sub': {}}
    elif cur is None: continue
    elif tag == 's_payload': cur['pay'] = t
    elif tag == 's_sub_b': cur['sub'][a] = [t, None]
    elif tag == 's_sub_e': cur['sub'][a][1] = t
    elif tag == 's_done':
        cur['done'] = t; cur['op'] = a
        if a == 50: bs.append(cur)
        cur = None
print('client jobs', len(jobs), 'server batches', len(bs))
n = min(len(jobs), len(bs))
# align from the end
jobs = jobs[len(jobs) - n:]; bs = bs[len(bs) - n:]
seg = {k: [] for k in ['submit->wake', 'wake->send_b', 'send(syscall)', 'send_b->srv_hdr(wake)', 'hdr->payload', 'payload->sub12_b', 'write x (12)', 'enqueue (31)', 'kernel+read (13)', 'sub13_e->done', 'done->c_resp_hdr(wake)', 'resp->exec_e', 'exec_e->host wait_e', 'job total (wake->exec_e)', 'submit->exec_e']}
for j, b in zip(jobs[skip:], bs[skip:]):
    us = lambda a, c: (c - a) / 1000.0
    s = b['sub']
    if 12 not in s or 31 not in s or 13 not in s: continue
    seg['submit->wake'].append(us(j['submit'], j['wake']))
    seg['wake->send_b'].append(us(j['wake'], j['send_b']))
    seg['send(syscall)'].append(us(j['send_b'], j['send_e']))
    seg['send_b->srv_hdr(wake)'].append(us(j['send_b'], b['hdr']))
    seg['hdr->payload'].append(us(b['hdr'], b['pay']))
    seg['payload->sub12_b'].append(us(b['pay'], s[12][0]))
    seg['write x (12)'].append(us(s[12][0], s[12][1]))
    seg['enqueue (31)'].append(us(s[31][0], s[31][1]))
    seg['kernel+read (13)'].append(us(s[13][0], s[13][1]))
    seg['sub13_e->done'].append(us(s[13][1], b['done']))
    seg['done->c_resp_hdr(wake)'].append(us(b['done'], j['resp']))
    seg['resp->exec_e'].append(us(j['resp'], j['exec_e']))
    if 'wait_e' in j: seg['exec_e->host wait_e'].append(us(j['exec_e'], j['wait_e']))
    seg['job total (wake->exec_e)'].append(us(j['wake'], j['exec_e']))
    seg['submit->exec_e'].append(us(j['submit'], j['exec_e']))
print('%-28s %9s %9s %9s %9s %9s  (us)' % ('segment', 'mean', 'p10', 'p50', 'p90', 'max'))
for k, v in seg.items():
    if not v: continue
    v = sorted(v)
    print('%-28s %9.0f %9.0f %9.0f %9.0f %9.0f' % (k, sum(v) / len(v), v[len(v) // 10], v[len(v) // 2], v[len(v) * 9 // 10], v[-1]))
