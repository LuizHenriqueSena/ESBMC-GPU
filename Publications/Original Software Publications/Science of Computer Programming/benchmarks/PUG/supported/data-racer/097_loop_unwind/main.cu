#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float *A)
{
    __shared__ float B[256];

    for(int i = 0; i < N*2; i ++) {
        B[i] = A[i];
    }
}

#endif
