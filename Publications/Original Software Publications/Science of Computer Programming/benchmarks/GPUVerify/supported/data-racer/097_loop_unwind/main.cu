//xfail:BOOGIE_ERROR
//--blockDim=512 --gridDim=64 --loop-unwind=2 --no-inline
//kernel.cu: error: possible write-write race on B

#include <cuda.h>

#include <stdio.h>
#include <assert.h>

#define N 2//512

extern "C" {

__global__ void helloCUDA(float *A)
{
    __shared__ float B[256];

    for(int i = 0; i < N*2; i ++) {
        B[i] = A[i];
    }
}

}

