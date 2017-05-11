#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "cuda.h"

//pass
//--blockDim=10 --gridDim=64 --no-inline

__global__ void foo() {

  __shared__ int A[10][10];
  A[threadIdx.y][threadIdx.x] = 2;
  __assert(A[threadIdx.y][threadIdx.x]!=2);
}

