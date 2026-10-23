__kernel void k(__global const float*a, __global float*o){ size_t i=get_global_id(0); vstore_half(a[i], i, (__global half*)o); }
