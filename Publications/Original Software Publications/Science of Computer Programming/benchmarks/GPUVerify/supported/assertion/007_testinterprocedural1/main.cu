//assertion
//--blockDim=64 --gridDim=64 --no-inline

#include "cuda.h"
#include <stdio.h>

#define N 1

__device__ void bar (int *p){

	int a =2;
	
	p = &a;
	__assert(*p == 2);
}

__global__ void foo (int* p, int* q){

	__requires(*p == 1)
	__requires(*q == 1)
    bar(p);

    bar(q);
	__assert(*p == 2);
    //*p = 23; *q = 23; // remove this comment to see that the __device__ function does not work
}

