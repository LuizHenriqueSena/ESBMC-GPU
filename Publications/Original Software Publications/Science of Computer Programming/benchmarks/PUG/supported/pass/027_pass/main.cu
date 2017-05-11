#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int *a, int *b, int *c){
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

#endif
