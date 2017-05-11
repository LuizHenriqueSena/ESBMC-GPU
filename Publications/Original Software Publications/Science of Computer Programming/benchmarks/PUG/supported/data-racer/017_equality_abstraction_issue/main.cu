#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int * A) {
    A[0] = 1;
    A[1] = 1;
    A[threadIdx.x] = 0;
//__assert(A[0] == 1 | A[1] == 1 | A[2] == 1);
}

#endif
