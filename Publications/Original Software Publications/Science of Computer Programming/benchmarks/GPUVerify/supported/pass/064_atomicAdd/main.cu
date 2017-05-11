//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda.h>
#include <stdio.h>

#define N 2 //1024

__global__ void definitions (int* A, unsigned int* B, unsigned long long int* C, float* D)
{
	atomicAdd(A,10);

	atomicAdd(B,10);

	atomicAdd(C,10);

	atomicAdd(D,10);

}

