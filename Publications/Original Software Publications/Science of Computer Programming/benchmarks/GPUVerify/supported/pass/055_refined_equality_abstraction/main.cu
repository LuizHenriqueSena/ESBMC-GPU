//pass
//--warp-sync=32 --blockDim=32 --gridDim=1 --equality-abstraction --no-inline
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define N 2//32

__global__ void foo(int * A, int * B) {
    A[threadIdx.x] = 1;
    volatile int x = A[threadIdx.x];
    B[threadIdx.x] = 1;
    volatile int y = A[threadIdx.x];
    __assert(x==y);
}

