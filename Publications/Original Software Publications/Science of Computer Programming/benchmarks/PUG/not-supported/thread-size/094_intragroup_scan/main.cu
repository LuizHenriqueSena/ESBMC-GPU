#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A)
{
	unsigned int tid = threadIdx.x;
	unsigned int lane = tid % 32;

	int temp [32] = {0};
	scan_warp(A);
	__syncthreads();

	if (lane == 31)	// ?????????
		temp[tid / 32] = A[tid];
	__syncthreads();

	if (tid / 32 == 0)
		scan_warp(temp);
	__syncthreads();

	A[tid] += temp[tid/32];

}

#endif
