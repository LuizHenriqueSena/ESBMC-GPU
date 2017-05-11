#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* A)
{
 
 //__assert(__all(threadIdx.x < blockDim.x));
	assert(threadIdx.x < blockDim.x);

}

#endif
