import json,sys
d=json.load(open(sys.argv[1])); S=d['samples']
# main thread = thread with name llama-bench and largest run
import collections
tot=collections.defaultdict(float)
for s in S:
    for tid,v in s['th'].items(): tot[(tid,v[0])]=max(tot[(tid,v[0])],v[4])
print('threads (tid,name,total run ms):'); [print(' ',k,round(v/1e6)) for k,v in sorted(tot.items(),key=lambda x:-x[1])[:8]]
main=max(tot,key=tot.get)[0]
# windows of 1s
w=1.0; cur=0; prev=None
rows=[]
for s in S:
    v=s['th'].get(main)
    if not v: continue
    rows.append((s['t'],v[4]/1e6,v[1],v[2],v[3]))
import math
i=0
while i<len(rows):
    t0=rows[i][0]; j=i
    while j<len(rows) and rows[j][0]-t0<w: j+=1
    if j>=len(rows): break
    dt=rows[j][0]-t0; dr=rows[j][1]-rows[i][1]; du=(rows[j][3]-rows[i][3])*10; ds=(rows[j][4]-rows[i][4])*10
    seg=[r[2] for r in rows[i:j]]
    print(f't={t0:6.1f} dt={dt:.2f} cpu={dr/dt/10:5.1f}% user={du/dt/10:5.1f}% sys={ds/dt/10:5.1f}% R-frac={seg.count("R")/len(seg):.2f}')
    i=j
