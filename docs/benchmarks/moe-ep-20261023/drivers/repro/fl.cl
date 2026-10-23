__kernel void k(__global const float*a, __global float*o){ size_t i=get_global_id(0); o[i]=fma(a[i],a[i],1.0f); }
