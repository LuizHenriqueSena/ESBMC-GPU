//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda.h>
#include <stdio.h>

#define N 2 //1024

__global__ void definitions (int* A, unsigned int* B, unsigned long long int* C)
{
  atomicOr(A,10);

  atomicOr(B,7);

  atomicOr(C,5);
}

