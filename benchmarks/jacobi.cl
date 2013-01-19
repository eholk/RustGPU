/* -*- c -*- */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define offset(x, i) (&((x)[(i)]))

__kernel void jacobi(__global void *foo, __global void *bar,
                     ulong N,
                     __global double *src,
                     __global double *dst)
{
    ulong i = get_global_id(0) + 1;
    ulong j = get_global_id(1) + 1;

    if(i >= N - 1 || j >= N - 1) { return; }

    double u = src[(i - 1) * N + j];
    double d = src[(i + 1) * N + j];
    double l = src[(i) * N + (j - 1)];
    double r = src[(i) * N + (j + 1)];

    dst[i * N + j] = (u + d + l + r) / 4.0;
}
