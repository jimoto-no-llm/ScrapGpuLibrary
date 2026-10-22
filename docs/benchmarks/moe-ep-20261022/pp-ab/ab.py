import json,os,statistics as st
L=os.path.expanduser('~/moe-ep/sglx/logs/')
def v(t):
    for l in open(L+t+'.out'):
        l=l.strip()
        if l.startswith('{'): d=json.loads(l); return d['avg_ts'],d['stddev_ts'],d['samples_ts']
a=[];b=[];diff=[]
for i in range(1,9):
    x=v('ppab_c1_%d'%i); y=v('ppab_C_%d'%i)
    a.append(x[0]); b.append(y[0]); diff.append(y[0]-x[0])
    print(i,'C1 %.1f±%.1f'%x[:2],'SGLX %.1f±%.1f'%y[:2],'diff %+.1f'%(y[0]-x[0]))
m=st.mean(diff); s=st.stdev(diff)
print('mean C1 %.1f SGLX %.1f diff %+.2f sd %.2f se %.2f t %.2f pos %d/8 rel %+.2f%%'%(st.mean(a),st.mean(b),m,s,s/8**.5,m/(s/8**.5),sum(d>0 for d in diff),100*m/st.mean(a)))
