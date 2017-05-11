#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int *A) {

	A[threadIdx.x] = bar(threadIdx.x);
}

#endif
