//fail
//--blockDim=64 --gridDim=64 --no-inline
#include "cuda.h"
#include <stdio.h>

#define N 1

__device__ void bar (int *p){

    int a = 0;

    p = &a;
}

__global__ void foo (int* p, int* q){

    if (*p > 10){
        bar(p);
        //*p = 23; // remove this comment to see that the __device__ function does not work
    }
    else {
        bar(q);
        //*q = 23; // remove this comment to see that the __device__ function does not work
    }
}

