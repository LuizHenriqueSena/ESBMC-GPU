#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, unsigned int* B)
{
	atomicSub(A,10);

	atomicSub(B,5);

}

#endif
