#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A)
{
	int tid = threadIdx.x;
	int warp = tid / 32;
	int* B = A + (warp*32);
	A[tid] = B[(tid + 1)%32];
}

#endif	
