#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* p, int* q){

    if (*p > 10){
        bar(p);
        //*p = 23; // remove this comment to see that the __device__ function does not work
    }
    else {
        bar(q);
        //*q = 23; // remove this comment to see that the __device__ function does not work
    }
}

#endif
