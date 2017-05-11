#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, unsigned int* B, unsigned long long int* C, float* D)
{
	atomicAdd(A,10);

	atomicAdd(B,10);

	atomicAdd(C,10);

	atomicAdd(D,10);

}

#endif
