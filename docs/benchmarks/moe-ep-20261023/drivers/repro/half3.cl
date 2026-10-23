__kernel void k(__global const ushort*a, __global float*o){ size_t i=get_global_id(0); o[i]=vload_half(i,(__global half*)a); }
