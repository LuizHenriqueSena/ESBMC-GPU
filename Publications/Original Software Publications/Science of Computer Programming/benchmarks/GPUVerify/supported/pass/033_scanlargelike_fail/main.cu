//fail
//--blockDim=32 --gridDim=64 --no-inline
#include <cuda.h>

#include <stdio.h>
#include <assert.h>

#define N 2//32

__device__ void f(float *odata, int* ai) {
    int thid = threadIdx.x;
    *ai = thid;
    odata[*ai] = 2*threadIdx.x;
}

__global__ void k(float *g_odata) {
    int ai;
    f(g_odata,&ai);
}

