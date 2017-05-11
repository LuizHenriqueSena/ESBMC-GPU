//data race
//--blockDim=512 --gridDim=1 --warp-sync=32 --no-inline

#include <cuda.h>

#include <stdio.h>
#include <assert.h>
#define N 4//512

__global__ void shuffle (int* A)
{
	int tid = threadIdx.x;
	int warp = tid / 2;//32;
	int* B = A + (warp*2);//32);
	A[tid] = B[(tid + 1)%2];//32];
}

