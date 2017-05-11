//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1 --warp-sync=16 --no-inline
//It should show only the values from B[0] to B[31], but it exceeds.

#include <cuda.h>

#include <stdio.h>

#define N 2//32//1024

__global__ void shuffle (int* A)
{
	int tid = threadIdx.x;
	int warp = tid / 32;
	int* B = A + (warp*32);
	A[tid] = B[(tid + 1)%32];
}

