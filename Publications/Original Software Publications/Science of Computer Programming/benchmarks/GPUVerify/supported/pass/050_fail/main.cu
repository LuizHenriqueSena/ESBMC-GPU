//fail
//--blockDim=2048 --gridDim=2 --no-inline
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define N 2//2048

__constant__ int A[4096];
__constant__ int B[3] = {0,1,2};

__global__ void kernel(int* x) {
  x[threadIdx.x] = A[threadIdx.x] + B[0]; //permanece constante por ser muito grande. N < 1024 não permanece
}

