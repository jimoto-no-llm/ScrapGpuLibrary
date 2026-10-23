__kernel void k(__global const float*a, __global half*o){ o[get_global_id(0)] = convert_half(a[get_global_id(0)]); }
