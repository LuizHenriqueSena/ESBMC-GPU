#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* p) {
    p[threadIdx.x] = 2;
    __syncthreads();
}

#endif
