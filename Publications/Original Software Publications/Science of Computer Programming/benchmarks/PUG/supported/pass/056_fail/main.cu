#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float *A, int sz) {
	assert(sz == blockDim.x);
	for(int i = threadIdx.x; i < 1*sz; i += sz) {
		A[i] *= 2.0f;
  }
}

#endif
