"""an_phase2.py *.txt.ev: using vkGetFenceStatus success events (exact GPU completion). Tokens split by >GAP ms idle.
two-device mode (-2): A = RX-ish first device phase (first compute submit -> first fence success), B = second device phase (first submit on last queue -> last success)
one-device mode (default): phase = first submit -> last fence success in token."""
import sys,statistics as st,re,os,gzip
GAP=4.0
def load(fn):
    ev=[]
    for l in (gzip.open(fn,'rt') if fn.endswith('.gz') else open(fn)):
        n,tid,a,b,arg=l.split()
        if n=='vkMapMemory': continue
        ev.append((n,float(a),float(b),arg))
    return ev
def tokens(ev):
    base=[e for e in ev if e[0]!='vkGetFenceStatus']; gfs=[e for e in ev if e[0]=='vkGetFenceStatus']
    toks=[];cur=[];pe=None
    for e in base:
        if pe is not None and e[1]-pe>GAP and cur: toks.append(cur);cur=[]
        cur.append(e);pe=e[2]
    toks.append(cur)
    starts=[t[0][1] for t in toks]
    import bisect
    for g in gfs:
        i=bisect.bisect_right(starts,g[1])-1
        if i>=0: toks[i].append(g)
    for t in toks: t.sort(key=lambda e:e[1])
    return [t for t in toks if sum(1 for e in t if e[0]=='vkQueueSubmit')>=2]
def one(fn):
    toks=tokens(load(fn)); steady=toks[max(2,len(toks)//4):-1]
    rows=[]
    for t in steady:
        subs=[e for e in t if e[0]=='vkQueueSubmit']; suc=[e for e in t if e[0]=='vkGetFenceStatus']
        if not subs or not suc: continue
        rows.append((suc[-1][1]-subs[0][1], len(subs), len(suc)))
    return st.mean(r[0] for r in rows), rows[0][1], len(rows)
def two(fn):
    toks=tokens(load(fn)); steady=toks[max(2,len(toks)//4):-1]
    rows=[]
    for t,nt in zip(steady,steady[1:]):
        subs=[e for e in t if e[0]=='vkQueueSubmit']; suc=[e for e in t if e[0]=='vkGetFenceStatus']
        if len(suc)<2: continue
        qs=[]; 
        for e in subs:
            if e[3] not in qs: qs.append(e[3])
        qb=qs[-1]; sb=[e for e in subs if e[3]==qb]
        # A: first submit on any queue other than the transfer queue (qs[0]) -> first success
        qa=[e for e in subs if e[3]!=qs[0] and e[3]!=qb]
        a0=(qa[0] if qa else subs[0])[1]
        A=suc[0][1]-a0; B=suc[-1][1]-sb[0][1]
        rows.append((A,B,sb[0][1]-suc[0][1],nt[0][1]-t[0][1]))
    m=lambda i: st.mean(r[i] for r in rows)
    return m(0),m(1),m(2),m(3),len(rows)
if __name__=='__main__':
    mode=sys.argv[1]
    for fn in sys.argv[2:]:
        tag=os.path.basename(fn).replace('.txt.ev.gz','').replace('.txt.ev','')
        out=open(re.sub(r'\.txt\.ev(\.gz)?$','.out',fn)).read(); ts=re.search(r'"avg_ts": ([\d.]+)',out)
        ngl=re.search(r'"n_gpu_layers": (\d+)',out)
        if mode=='2':
            A,B,H,P,n=two(fn); print(f'{tag:12s} tg={float(ts[1]):6.2f}  period={P:6.2f}  A(first dev GPU)={A:6.2f}  gap={H:5.2f}  B(second dev GPU)={B:6.2f}  host-rest={P-A-H-B:5.2f}  n={n}')
        else:
            A,ns,n=one(fn); print(f'{tag:14s} ngl={ngl[1] if ngl else ""} tg={float(ts[1]):6.3f}  gpu phase={A:6.2f} ms  submits/token={ns} n={n}')
