#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float * A, int x) {

    if(threadIdx.x == 0) {
        A[threadIdx.x + x] = threadIdx.x; //A[1] = 0;
    }

    if(threadIdx.x == 1) {
        A[threadIdx.x] = threadIdx.x; //A[1] = 1;
   }
}

#endif
