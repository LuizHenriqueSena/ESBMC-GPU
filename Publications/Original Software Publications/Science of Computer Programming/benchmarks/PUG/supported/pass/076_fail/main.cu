#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float* A) {

  if(blockIdx.x == 0)
	  A[threadIdx.x] = 50.f;
}

#endif
