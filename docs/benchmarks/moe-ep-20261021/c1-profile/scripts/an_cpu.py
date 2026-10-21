import json,sys,glob,re
rows={}
for f in sorted(glob.glob(sys.argv[1]+'/cpu_t*_n*.json')):
    m=re.search(r'cpu_t(\d)_n(\d+)\.json',f); t,n=int(m[1]),int(m[2])
    d=json.load(open(f)); out=open(f.replace('.json','.out')).read()
    ts=float(re.search(r'"avg_ts": ([\d.]+)',out)[1])
    rows.setdefault(t,{})[n]=(d,ts)
for t in sorted(rows):
    ns=sorted(rows[t]); a,b=ns[0],ns[-1]
    da,ta=rows[t][a]; db,tb=rows[t][b]
    print(f'== -t {t}: n={a} {ta:.2f} t/s, n={b} {tb:.2f} t/s')
    for n in ns:
        d,ts=rows[t][n]; print(f'  n={n}: wall {d["wall"]:.2f}s user {d["utime"]:.2f} sys {d["stime"]:.2f}  nvcsw {d["nvcsw"]} nivcsw {d["nivcsw"]}')
    dn=b-a
    print(f'  marginal per token (n{b}-n{a}): user {(db["utime"]-da["utime"])/dn*1000:.2f} ms sys {(db["stime"]-da["stime"])/dn*1000:.2f} ms wall {(db["wall"]-da["wall"])/dn*1000:.2f} ms')
    # per thread: run_ns delta by thread name, matching by sorted tid order
    def agg(d):
        r={}
        for tid,v in d['last'].items():
            r.setdefault(v[0],[]).append((int(tid),v[4]/1e6,v[2],v[3]))
        return r
    A,B=agg(da),agg(db)
    for name in B:
        sb=sorted(B[name]); sa=sorted(A.get(name,[]))
        for i,x in enumerate(sb):
            ra=sa[i][1] if i<len(sa) else 0
            print(f'    {name:16s} tid#{i} run_ms n{b}={x[1]:.0f} n{a}={ra:.0f} d/tok={(x[1]-ra)/dn:.2f} ms')
