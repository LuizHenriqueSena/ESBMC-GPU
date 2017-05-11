//fail: data-race, all the threads write on A[0]

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>

#include <sm_atomic_functions.h>

#define N 2

__global__ void race_test (unsigned int* i, int* A)
{
	int tid = threadIdx.x;
	int j = atomicAdd(i,0);
  	A[j] = tid;
}

