#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
__global__ void vadd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("%s failed: %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)
int main() {
  int nd = 0; CK(hipGetDeviceCount(&nd));
  printf("devices: %d\n", nd);
  hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, 0));
  printf("%s arch=%s mem=%.2f GiB cu=%d\n", p.name, p.gcnArchName, p.totalGlobalMem / 1073741824.0, p.multiProcessorCount);
  const int n = 1 << 24;
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; i++) { a[i] = i; b[i] = 2 * i; }
  float *da, *db, *dc;
  CK(hipMalloc(&da, n * 4)); CK(hipMalloc(&db, n * 4)); CK(hipMalloc(&dc, n * 4));
  CK(hipMemcpy(da, a.data(), n * 4, hipMemcpyHostToDevice)); CK(hipMemcpy(db, b.data(), n * 4, hipMemcpyHostToDevice));
  vadd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  CK(hipGetLastError()); CK(hipDeviceSynchronize());
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0); for (int r = 0; r < 20; r++) vadd<<<(n + 255) / 256, 256>>>(da, db, dc, n); hipEventRecord(e1); hipEventSynchronize(e1);
  float ms; hipEventElapsedTime(&ms, e0, e1);
  CK(hipMemcpy(c.data(), dc, n * 4, hipMemcpyDeviceToHost));
  int bad = 0; for (int i = 0; i < n; i++) if (c[i] != 3.0f * i) bad++;
  printf("bad=%d  vadd %.3f ms  %.1f GB/s\n", bad, ms / 20, 3.0 * n * 4 / (ms / 20 * 1e-3) / 1e9);
  return bad != 0;
}
