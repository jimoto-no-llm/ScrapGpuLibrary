#include <string.h>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
/* usage: clbuild file.cl [platform-substring] : build the program for the first GPU device of the matching platform, print the build log */
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb"); if(!f) return 2; fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
  char*src=malloc(n+1); fread(src,1,n,f); src[n]=0;
  cl_platform_id P[8]; cl_uint np; clGetPlatformIDs(8,P,&np);
  for(cl_uint i=0;i<np;i++){ char nm[128]; clGetPlatformInfo(P[i],CL_PLATFORM_NAME,128,nm,0);
    if(argc>2 && !strstr(nm,argv[2])) continue;
    cl_device_id d; cl_uint nd; if(clGetDeviceIDs(P[i],CL_DEVICE_TYPE_GPU,1,&d,&nd)!=CL_SUCCESS||!nd) continue;
    char dn[128]; clGetDeviceInfo(d,CL_DEVICE_NAME,128,dn,0); printf("platform %s device %s\n",nm,dn); fflush(stdout);
    cl_int e; cl_context c=clCreateContext(0,1,&d,0,0,&e); cl_program p=clCreateProgramWithSource(c,1,(const char**)&src,0,&e);
    e=clBuildProgram(p,1,&d,"",0,0); printf("build rc=%d\n",e);
    char log[8192]; clGetProgramBuildInfo(p,d,CL_PROGRAM_BUILD_LOG,8192,log,0); printf("log: %s\n",log); return e!=0;
  } printf("no device\n"); return 3; }
