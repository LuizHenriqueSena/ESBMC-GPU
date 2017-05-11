#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(double *H) {

	int idx = index (threadIdx.x,threadIdx.y,threadIdx.z);

	H[idx] = C[threadIdx.x][threadIdx.y][threadIdx.z];
}

#endif
