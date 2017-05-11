#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A)
{
	int tid = threadIdx.x;
	unsigned int lane = tid & 31;

	if (lane >= 1) A[tid] = A[tid - 1] + A[tid];
	if (lane >= 2) A[tid] = A[tid - 2] + A[tid];
	if (lane >= 4) A[tid] = A[tid - 4] + A[tid];
	if (lane >= 8) A[tid] = A[tid - 8] + A[tid];
	if (lane >= 16) A[tid] = A[tid - 16] + A[tid];
}

#endif
