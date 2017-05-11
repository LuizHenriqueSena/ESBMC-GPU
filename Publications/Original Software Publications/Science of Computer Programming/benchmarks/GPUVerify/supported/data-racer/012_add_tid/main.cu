//xfail:BOOGIE_ERROR:data race
//--blockDim=2 --gridDim=1 --no-inline

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>

#include <sm_atomic_functions.h>

#define N 2

__global__ void race_test (unsigned int* i, int* A)
{
	int tid = threadIdx.x;
	int j = atomicAdd(i,tid);
	A[j] = tid;
}

