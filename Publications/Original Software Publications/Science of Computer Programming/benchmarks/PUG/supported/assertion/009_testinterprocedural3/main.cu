#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* p, int* q){

    __shared__ int sharedArr  [100];

    __shared__ int sharedArr2 [50];

    bar(p);

    baz (sharedArr);

    bar(q);

    if (*q){
        baz(sharedArr2);
    }

    //*p = 23; *q = 23; // remove this comment to see that the __device__ function does not work
}

#endif
