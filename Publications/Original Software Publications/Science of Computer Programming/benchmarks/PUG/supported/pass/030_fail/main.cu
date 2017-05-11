#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float* A, float c) {

	A[threadIdx.x == 0 ? 1 : 2*threadIdx.x] = c;

}

#endif
