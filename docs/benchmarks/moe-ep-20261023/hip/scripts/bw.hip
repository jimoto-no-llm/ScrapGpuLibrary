// host<->device copy bandwidth of the RX 6400 through HIP: pageable, pinned (hipHostMalloc), hipHostRegister'ed memory; 256 MiB blocks
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s failed: %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  const size_t n = 256ull << 20;
  void *d; CK(hipMalloc(&d, n));
  char *pg = (char *) malloc(n); memset(pg, 1, n);
  char *pin; CK(hipHostMalloc((void **) &pin, n)); memset(pin, 1, n);
  char *reg = (char *) malloc(n); memset(reg, 1, n); CK(hipHostRegister(reg, n, hipHostRegisterPortable));
  struct { const char *name; char *p; } t[] = {{"pageable", pg}, {"pinned(hipHostMalloc)", pin}, {"registered", reg}};
  for (auto &x : t) {
    for (int dir = 0; dir < 2; dir++) {
      CK(hipMemcpy(dir ? (void *) x.p : d, dir ? d : (void *) x.p, n, dir ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice));
      double t0 = now();
      for (int r = 0; r < 4; r++) CK(hipMemcpyAsync(dir ? (void *) x.p : d, dir ? d : (void *) x.p, n, dir ? hipMemcpyDeviceToHost : hipMemcpyHostToDevice));
      CK(hipDeviceSynchronize());
      double s = now() - t0;
      printf("%-22s %s: %.2f GB/s\n", x.name, dir ? "D2H" : "H2D", 4.0 * n / s / 1e9);
    }
  }
  return 0;
}
