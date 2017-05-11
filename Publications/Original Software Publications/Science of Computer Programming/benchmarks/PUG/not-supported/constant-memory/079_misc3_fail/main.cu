#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__constant__ int A[N] = {0, 1, 2, 3};

__global__ void kernel(int *B) {

//	assert(A[0]==0);
	A[threadIdx.x] = B[threadIdx.x];
//	assert(A[0]==0); // the constant memory was modified!!!
	__syncthreads();

	B[threadIdx.x] = A[threadIdx.x];

}

#endif
