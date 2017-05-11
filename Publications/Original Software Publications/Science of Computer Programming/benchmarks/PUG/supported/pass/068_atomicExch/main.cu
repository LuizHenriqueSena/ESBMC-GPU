#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, unsigned int* B, unsigned long long int* C, float* D)
{
	atomicExch(A,10);

	atomicExch(B,100);

	atomicExch(C,20);

	atomicExch(D,200.0);
}

#endif
