import re,sys,statistics as st,collections
fn=sys.argv[1]
blocks=[b for b in open(fn).read().split('----------------') if 'Vulkan Timings' in b]
def parse(b):
    d={}
    for l in b.splitlines():
        m=re.match(r'(.+?): (\d+) x ([\d.]+) us = ([\d.]+) us',l)
        if m:
            k=m[1] if 'q6_K' not in m[1] else 'OUTPUT '+m[1]
            d[k]=(int(m[2]),float(m[4]))
        m=re.match(r'Total time: ([\d.]+) us',l)
        if m: d['__total']=float(m[1])
    return d
bl=[parse(b) for b in blocks]
isW=lambda d: any('q6_K' in k for k in d)
W=[d for d in bl[4:] if isW(d)]; R=[d for d in bl[4:] if not isW(d)]
def cat(k):
    if k.startswith('OUTPUT'): return 'output q6_K mat-vec'
    if 'MUL_MAT_ID' in k: return 'MUL_MAT_ID (expert up/gate/down)'
    if 'MUL_MAT' in k: return 'mat-vec (q4_0 conv/attn/dense, f16, f32)'
    return 'other small ops (norm/add/mul/glu/copy/conv...)'
for name,L in ((sys.argv[2],R),(sys.argv[3],W)):
    agg=collections.defaultdict(list); tot=[]
    for d in L:
        c=collections.defaultdict(float)
        for k,v in d.items():
            if k!='__total': c[cat(k)]+=v[1]
        for k in c: agg[k].append(c[k])
        tot.append(d['__total'])
    print(f'{name}: blocks={len(L)} total {st.mean(tot)/1000:.2f} ms')
    for k,v in sorted(agg.items(),key=lambda x:-st.mean(x[1])): print(f'   {k:50s} {st.mean(v)/1000:6.2f} ms')
