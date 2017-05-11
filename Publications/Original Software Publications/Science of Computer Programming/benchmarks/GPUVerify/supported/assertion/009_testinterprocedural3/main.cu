//fail
//--blockDim=64 --gridDim=64 --no-inline
#include "cuda.h"
#include <stdio.h>

#define N 1

__device__ void baz (int p []){
    int a;

    p = &a;
}

__device__ void bar (int *p){

    int a = 2;

    p = &a;
}


__global__ void foo (int* p, int* q){

	__requires (*q == 1)
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

