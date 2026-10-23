// LD_PRELOAD CPU-time sampling profiler (no perf/gdb on tb250). SIGPROF (ITIMER_PROF, process CPU time) -> backtrace().
// Output SPROF_OUT: one line per sample "tid frame0 frame1 ..." with frames as module+0xoffset.
#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <unistd.h>
#define MAXS 300000
#define DEPTH 24
static void *buf[MAXS][DEPTH]; static short nfr[MAXS]; static int tids[MAXS]; static volatile int ns;
static void h(int s){ int i=__atomic_fetch_add(&ns,1,__ATOMIC_RELAXED); if(i>=MAXS) return;
  nfr[i]=(short)backtrace(buf[i],DEPTH); tids[i]=(int)syscall(SYS_gettid); }
__attribute__((constructor)) static void init(void){
  void*tmp[4]; backtrace(tmp,4);
  const char*e=getenv("SPROF_HZ"); int hz=e?atoi(e):1000; if(!getenv("SPROF_OUT")) return;
  struct sigaction sa; memset(&sa,0,sizeof sa); sa.sa_handler=h; sa.sa_flags=SA_RESTART; sigaction(SIGPROF,&sa,0);
  struct itimerval it={{0,1000000/hz},{0,1000000/hz}}; setitimer(ITIMER_PROF,&it,0); }
typedef struct { unsigned long long a,b,off; char path[300]; } M; static M maps[2000]; static int nm;
__attribute__((destructor)) static void fini(void){
  struct itimerval z={{0,0},{0,0}}; setitimer(ITIMER_PROF,&z,0);
  const char*o=getenv("SPROF_OUT"); if(!o) return;
  FILE*m=fopen("/proc/self/maps","r"); char l[1024];
  while(m && fgets(l,sizeof l,m) && nm<2000){ unsigned long long a,b,off; char perm[8],path[300]=""; if(sscanf(l,"%llx-%llx %7s %llx %*s %*s %299s",&a,&b,perm,&off,path)>=4){ maps[nm++]=(M){a,b,off,""}; strncpy(maps[nm-1].path,path,299);} }
  if(m) fclose(m);
  FILE*f=fopen(o,"w"); int n=ns<MAXS?ns:MAXS;
  for(int i=0;i<n;i++){ fprintf(f,"%d",tids[i]);
    for(int k=0;k<nfr[i];k++){ unsigned long long p=(unsigned long long)buf[i][k]; int j; for(j=0;j<nm;j++) if(p>=maps[j].a&&p<maps[j].b) break;
      if(j<nm) fprintf(f," %s+0x%llx",maps[j].path[0]?maps[j].path:"anon",p-maps[j].a+maps[j].off); else fprintf(f," ?+0x%llx",p);} fputc('\n',f);} fclose(f); }
