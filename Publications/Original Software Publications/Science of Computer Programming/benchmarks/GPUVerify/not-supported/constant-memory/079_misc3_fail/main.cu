//xfail:BOOGIE_ERROR
//possible attempt to modify constant memory
//You can modify the values of the constants, uncomment the lines 14 and 16 to analyze this case.

#include <cuda.h>

#include <stdio.h>
#include <assert.h>

#define N 2//1024

__constant__ int A[N] = {0, 1, 2, 3};

__global__ void foo(int *B) {

//	assert(A[0]==0);
	A[threadIdx.x] = B[threadIdx.x];
//	assert(A[0]==0); // the constant memory was modified!!!
	__syncthreads();

	B[threadIdx.x] = A[threadIdx.x];

}

