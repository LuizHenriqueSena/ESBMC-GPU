#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float *x, float y) {
	x[threadIdx.x] = __exp10f(y);	// pow(10,y), in this  case pow(10,2) = 100
}

#endif
