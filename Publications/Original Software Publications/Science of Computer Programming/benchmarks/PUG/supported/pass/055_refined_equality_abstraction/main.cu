#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int * A, int * B) {
    A[threadIdx.x] = 1;
    volatile int x = A[threadIdx.x];
    B[threadIdx.x] = 1;
    volatile int y = A[threadIdx.x];
    assert(x==y);
}

#endif
