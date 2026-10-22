import sys
ev=[l.split() for l in open(sys.argv[1])]
ev=[(e[0],int(e[1]),float(e[2]),float(e[3]),e[4]) for e in ev]
npass=int(sys.argv[2]) if len(sys.argv)>2 else 1
# group queue submits per queue, gaps >30ms start a new group; print the last window
subs=[e for e in ev if e[0]=="vkQueueSubmit"]
groups=[]
for e in subs:
    k=e[4][-6:]
    if groups and groups[-1][0]==k and e[2]-groups[-1][2]<30: groups[-1][2]=e[2]; groups[-1][3]+=1
    else: groups.append([k,e[2],e[2],1])
big=[e for e in ev if e[0]=="vkWaitForFences" and e[3]-e[2]>20]
gp=[(g[1],"SUB q=%s .. %.1f n=%d"%(g[0],g[2],g[3])) for g in groups]
wp=[(e[2],"WAIT dev=%s until %.1f (%.1f ms)"%(e[4][-6:],e[3],e[3]-e[2])) for e in big]
allp=sorted(gp+wp)
# last window: from the start of the last npass passes
tstart=big[-3*npass][2]-300 if len(big)>=3*npass else 0
for t,s in allp:
    if t>=tstart and (s.startswith("WAIT") or "n=1 " not in s+" "): print("%.1f %s"%(t,s))
