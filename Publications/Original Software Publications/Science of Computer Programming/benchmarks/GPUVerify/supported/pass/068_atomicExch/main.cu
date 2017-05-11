//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda.h>
#include <stdio.h>

#define N 2 //1024

__global__ void definitions (int* A, unsigned int* B, unsigned long long int* C, float* D)
{
	atomicExch(A,10);

	atomicExch(B,100);

	atomicExch(C,20);

	atomicExch(D,200.0);
}

