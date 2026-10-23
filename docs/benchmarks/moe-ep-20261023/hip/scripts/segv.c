// LD_PRELOAD: print the faulting address, RIP, the module and the instruction bytes on SIGSEGV/SIGILL/SIGBUS
#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>
#include <execinfo.h>
static void h(int sig, siginfo_t *si, void *uc_) {
  ucontext_t *uc = uc_;
  unsigned long rip = uc->uc_mcontext.gregs[REG_RIP];
  fprintf(stderr, "SIG%d addr=%p rip=%#lx bytes:", sig, si->si_addr, rip);
  unsigned char *p = (unsigned char *)rip;
  for (int i = 0; i < 12; i++) fprintf(stderr, " %02x", p[i]);
  fprintf(stderr, "\n");
  FILE *f = fopen("/proc/self/maps", "r"); char l[512];
  while (f && fgets(l, sizeof l, f)) { unsigned long a, b; if (sscanf(l, "%lx-%lx", &a, &b) == 2 && rip >= a && rip < b) fprintf(stderr, "MAP %s", l); }
  void *bt[30]; int n = backtrace(bt, 30); backtrace_symbols_fd(bt, n, 2);
  _exit(99);
}
__attribute__((constructor)) static void init(void) {
  struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = h; sa.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, 0); sigaction(SIGILL, &sa, 0); sigaction(SIGBUS, &sa, 0);
}
