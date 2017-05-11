#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__constant__ int global_constant[N]; //= {0, 1, 2, 3};

__global__ void kernel(int *in) {

	global_constant[threadIdx.x] = in[threadIdx.x];

	__syncthreads();

	in[threadIdx.x] = global_constant[threadIdx.x];

}

#endif
