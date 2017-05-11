#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel()
{
	__shared__ int A[8];
	A[0] = threadIdx.x;
}

#endif
