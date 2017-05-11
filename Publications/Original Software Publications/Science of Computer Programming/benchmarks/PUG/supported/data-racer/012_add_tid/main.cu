#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (unsigned int* i, int* A)
{
	int tid = threadIdx.x;
	int j = atomicAdd(i,tid);
	A[j] = tid;
}

#endif
