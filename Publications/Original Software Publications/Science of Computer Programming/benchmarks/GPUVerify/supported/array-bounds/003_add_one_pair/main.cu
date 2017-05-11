//pass

#include <cuda.h>

#include <assert.h>

#define N 2

__global__ void race_test (unsigned int* i, int* A)
{
  int tid = threadIdx.x;
  int j = atomicAdd(i,1);
  A[j] = tid;
}

