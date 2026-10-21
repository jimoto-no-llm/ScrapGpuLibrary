// LD_PRELOAD shim: times selected Vulkan entry points (measurement only; forwards every call unchanged).
// Intercepts vkGetInstanceProcAddr/vkGetDeviceProcAddr and returns timing wrappers.
// Output at exit (VKSHIM_OUT): per-function count / total ms / max ms, and (VKSHIM_EVENTS=1) an event log.
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>

static inline long long now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000000000LL+t.tv_nsec; }
#define MAXF 64
typedef struct { const char*name; PFN_vkVoidFunction real; PFN_vkVoidFunction wrap; long long cnt, tot, max; int ev; } F;
static F fs[MAXF]; static int nf;
typedef struct { int f; int tid; long long t0, t1; unsigned long long arg; } Ev;
static Ev *evs; static long nev, capev; static int want_ev; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static long long tstart;
static void rec(int f, long long t0, long long t1, unsigned long long arg){
  F*x=&fs[f]; __atomic_fetch_add(&x->cnt,1,__ATOMIC_RELAXED); __atomic_fetch_add(&x->tot,t1-t0,__ATOMIC_RELAXED);
  if(t1-t0 > x->max) x->max=t1-t0;
  if(want_ev && x->ev){ pthread_mutex_lock(&mu); if(nev<capev){ evs[nev++] = (Ev){f,(int)syscall(SYS_gettid),t0,t1,arg}; } pthread_mutex_unlock(&mu);} }

#define WRAP(RET, NAME, PARAMS, CALLARGS, ARG) \
 static int id_##NAME; \
 static RET w_##NAME PARAMS { long long a=now_ns(); RET rv__ = ((RET (*) PARAMS) fs[id_##NAME].real) CALLARGS; rec(id_##NAME,a,now_ns(),(ARG)); return rv__; }
#define WRAPV(NAME, PARAMS, CALLARGS, ARG) \
 static int id_##NAME; \
 static void w_##NAME PARAMS { long long a=now_ns(); ((void (*) PARAMS) fs[id_##NAME].real) CALLARGS; rec(id_##NAME,a,now_ns(),(ARG)); }

WRAP(VkResult, vkQueueSubmit, (VkQueue q,uint32_t n,const VkSubmitInfo*s,VkFence f),(q,n,s,f), (unsigned long long)q)
WRAP(VkResult, vkQueueSubmit2, (VkQueue q,uint32_t n,const VkSubmitInfo2*s,VkFence f),(q,n,s,f), (unsigned long long)q)
WRAP(VkResult, vkWaitForFences, (VkDevice d,uint32_t n,const VkFence*f,VkBool32 w,uint64_t t),(d,n,f,w,t), (unsigned long long)d)
static int id_vkGetFenceStatus;
static VkResult w_vkGetFenceStatus(VkDevice d, VkFence f){ long long a=now_ns(); VkResult r=((VkResult(*)(VkDevice,VkFence))fs[id_vkGetFenceStatus].real)(d,f); long long b=now_ns();
  if(r==VK_SUCCESS) rec(id_vkGetFenceStatus,a,b,(unsigned long long)f); else { __atomic_fetch_add(&fs[id_vkGetFenceStatus+0].max,0,__ATOMIC_RELAXED); } return r; }
WRAP(VkResult, vkWaitSemaphores, (VkDevice d,const VkSemaphoreWaitInfo*i,uint64_t t),(d,i,t), (unsigned long long)d)
WRAP(VkResult, vkQueueWaitIdle, (VkQueue q),(q), (unsigned long long)q)
WRAP(VkResult, vkDeviceWaitIdle, (VkDevice d),(d), (unsigned long long)d)
WRAP(VkResult, vkResetFences, (VkDevice d,uint32_t n,const VkFence*f),(d,n,f), 0)
WRAP(VkResult, vkBeginCommandBuffer, (VkCommandBuffer c,const VkCommandBufferBeginInfo*i),(c,i), 0)
WRAP(VkResult, vkEndCommandBuffer, (VkCommandBuffer c),(c), 0)
WRAP(VkResult, vkAllocateCommandBuffers, (VkDevice d,const VkCommandBufferAllocateInfo*i,VkCommandBuffer*c),(d,i,c), 0)
WRAP(VkResult, vkResetCommandPool, (VkDevice d,VkCommandPool p,VkCommandPoolResetFlags f),(d,p,f), 0)
WRAP(VkResult, vkResetCommandBuffer, (VkCommandBuffer c,VkCommandBufferResetFlags f),(c,f), 0)
WRAP(VkResult, vkAllocateDescriptorSets, (VkDevice d,const VkDescriptorSetAllocateInfo*i,VkDescriptorSet*s),(d,i,s), 0)
WRAP(VkResult, vkResetDescriptorPool, (VkDevice d,VkDescriptorPool p,VkDescriptorPoolResetFlags f),(d,p,f), 0)
WRAPV(vkUpdateDescriptorSets, (VkDevice d,uint32_t nw,const VkWriteDescriptorSet*w,uint32_t nc,const VkCopyDescriptorSet*c),(d,nw,w,nc,c), 0)
WRAPV(vkCmdDispatch, (VkCommandBuffer c,uint32_t x,uint32_t y,uint32_t z),(c,x,y,z), 0)
WRAPV(vkCmdBindPipeline, (VkCommandBuffer c,VkPipelineBindPoint b,VkPipeline p),(c,b,p), 0)
WRAPV(vkCmdBindDescriptorSets, (VkCommandBuffer c,VkPipelineBindPoint b,VkPipelineLayout l,uint32_t fs_,uint32_t n,const VkDescriptorSet*s,uint32_t dn,const uint32_t*d),(c,b,l,fs_,n,s,dn,d), 0)
WRAPV(vkCmdPushConstants, (VkCommandBuffer c,VkPipelineLayout l,VkShaderStageFlags s,uint32_t o,uint32_t sz,const void*v),(c,l,s,o,sz,v), 0)
WRAPV(vkCmdPipelineBarrier, (VkCommandBuffer c,VkPipelineStageFlags s,VkPipelineStageFlags d,VkDependencyFlags f,uint32_t mn,const VkMemoryBarrier*m,uint32_t bn,const VkBufferMemoryBarrier*b,uint32_t in,const VkImageMemoryBarrier*i),(c,s,d,f,mn,m,bn,b,in,i), 0)
WRAPV(vkCmdCopyBuffer, (VkCommandBuffer c,VkBuffer s,VkBuffer d,uint32_t n,const VkBufferCopy*r),(c,s,d,n,r), 0)
WRAPV(vkCmdFillBuffer, (VkCommandBuffer c,VkBuffer b,VkDeviceSize o,VkDeviceSize s,uint32_t d),(c,b,o,s,d), 0)
WRAPV(vkCmdResetQueryPool, (VkCommandBuffer c,VkQueryPool p,uint32_t f,uint32_t n),(c,p,f,n), 0)
WRAP(VkResult, vkMapMemory, (VkDevice d,VkDeviceMemory m,VkDeviceSize o,VkDeviceSize s,VkMemoryMapFlags f,void**pp),(d,m,o,s,f,pp), 0)
WRAP(VkResult, vkFlushMappedMemoryRanges, (VkDevice d,uint32_t n,const VkMappedMemoryRange*r),(d,n,r), 0)
WRAP(VkResult, vkInvalidateMappedMemoryRanges, (VkDevice d,uint32_t n,const VkMappedMemoryRange*r),(d,n,r), 0)

#define REG(NAME) do{ id_##NAME = nf; fs[nf++] = (F){#NAME,0,(PFN_vkVoidFunction)w_##NAME,0,0,0,!strncmp(#NAME,"vkCmd",5)?0:(!strcmp(#NAME,"vkUpdateDescriptorSets")?0:1)}; }while(0)
static void reg_all(void){
  if(nf) return;
  REG(vkQueueSubmit);REG(vkGetFenceStatus);REG(vkQueueSubmit2);REG(vkWaitForFences);REG(vkWaitSemaphores);REG(vkQueueWaitIdle);REG(vkDeviceWaitIdle);REG(vkResetFences);
  REG(vkBeginCommandBuffer);REG(vkEndCommandBuffer);REG(vkAllocateCommandBuffers);REG(vkResetCommandPool);REG(vkResetCommandBuffer);
  REG(vkAllocateDescriptorSets);REG(vkResetDescriptorPool);REG(vkUpdateDescriptorSets);
  REG(vkCmdDispatch);REG(vkCmdBindPipeline);REG(vkCmdBindDescriptorSets);REG(vkCmdPushConstants);REG(vkCmdPipelineBarrier);REG(vkCmdCopyBuffer);REG(vkCmdFillBuffer);REG(vkCmdResetQueryPool);
  REG(vkMapMemory);REG(vkFlushMappedMemoryRanges);REG(vkInvalidateMappedMemoryRanges);
  want_ev = getenv("VKSHIM_EVENTS") != NULL;
  if(want_ev){ capev = 4000000; evs = malloc(sizeof(Ev)*capev); }
  tstart = now_ns();
}
static PFN_vkVoidFunction hook(const char*name, PFN_vkVoidFunction real){
  reg_all();
  if(!real || !name) return real;
  for(int i=0;i<nf;i++) if(!strcmp(fs[i].name,name)){ if(!fs[i].real) fs[i].real = real; return fs[i].wrap; }
  return real;
}
static void* vkh(void){ static void*h; if(!h) h=dlopen("libvulkan.so.1",RTLD_LAZY); return h; }
typedef PFN_vkVoidFunction (*GIPA)(VkInstance,const char*);
typedef PFN_vkVoidFunction (*GDPA)(VkDevice,const char*);
PFN_vkVoidFunction vkGetInstanceProcAddr(VkInstance i,const char*n){
  static GIPA real; if(!real) real=(GIPA)dlsym(vkh(),"vkGetInstanceProcAddr");
  return hook(n, real(i,n)); }
PFN_vkVoidFunction vkGetDeviceProcAddr(VkDevice d,const char*n){
  static GDPA real; if(!real) real=(GDPA)dlsym(vkh(),"vkGetDeviceProcAddr");
  return hook(n, real(d,n)); }
__attribute__((destructor)) static void dump(void){
  const char*o=getenv("VKSHIM_OUT"); if(!o||!nf) return;
  FILE*f=fopen(o,"w"); if(!f) return;
  fprintf(f,"# func count total_ms max_ms\n");
  for(int i=0;i<nf;i++) if(fs[i].cnt) fprintf(f,"%s %lld %.3f %.3f\n",fs[i].name,fs[i].cnt,fs[i].tot/1e6,fs[i].max/1e6);
  fclose(f);
  if(want_ev){ char p[512]; snprintf(p,sizeof p,"%s.ev",o); f=fopen(p,"w");
    for(long i=0;i<nev;i++) fprintf(f,"%s %d %.6f %.6f %llx\n",fs[evs[i].f].name,evs[i].tid,(evs[i].t0-tstart)/1e6,(evs[i].t1-tstart)/1e6,evs[i].arg);
    fclose(f);} }
