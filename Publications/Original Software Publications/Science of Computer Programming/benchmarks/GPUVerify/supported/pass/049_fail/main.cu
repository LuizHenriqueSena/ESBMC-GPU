#include <stdio.h>
#include <assert.h>
#include "cuda.h"

#define N 2

__global__ void foo(int* p) {
    p[threadIdx.x] = 2;
    __syncthreads();
}

