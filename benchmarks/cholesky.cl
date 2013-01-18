/* -*- c -*- */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef unsigned long uint64_t;

__kernel void update_kk(__global void *foo, __global void *bar,
                        uint64_t N, __global double *data, uint64_t k)
{
    data[k * N + k] = sqrt(data[k * N + k]);
}

__kernel void update_k(__global void *foo, __global void * bar,
                        uint64_t N, __global double *data, uint64_t k)
{
    int i = get_global_id(0);

    if(i > k && i < N) {
        double Akk = data[k * N + k];
        data[i * N + k] = data[i * N + k] / Akk;
        
        // zero out the top too.
        data[k * N + i] = 0;
    }
}

__kernel void update_block(__global void *foo, __global void * bar,
                        uint64_t N, __global double *data, uint64_t k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if(i <= k || j <= k) return;
    if(i >= N || j >  i) return;

    double Aik = data[i * N + k];
    double Ajk = data[j * N + k];
    double Aij = data[i * N + j];

    data[i * N + j] = Aij - Aik * Ajk;
}
