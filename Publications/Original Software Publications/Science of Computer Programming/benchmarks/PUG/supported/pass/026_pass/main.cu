#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

#endif
