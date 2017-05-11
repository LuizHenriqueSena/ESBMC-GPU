#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "cutil.h"

__global__ void kernel(int	*p, int *ptr_a) {

	ptr_a = p + threadIdx.x;

//	assert(*ptr_a == 2);
}

#endif
