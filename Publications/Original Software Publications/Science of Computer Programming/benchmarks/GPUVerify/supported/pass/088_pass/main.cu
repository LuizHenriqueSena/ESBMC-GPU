//pass
//--blockDim=1024 --gridDim=1024 --no-inline

#include <stdio.h>
#include <assert.h>

#include <cuda.h>

#include <math_functions.h>

#define DIM 2 //1024 in the future
#define N 2//DIM*DIM

__global__ void mul24_test (int* A, int* B)
{
  int idxa          = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int idxb = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  A[idxa] = idxa;
  B[idxb] = idxa;
}

