//pass
//--blockDim=64 --gridDim=64 --no-inline

#include <cuda.h>
#include <stdio.h>
#define N 2

__device__ void bar(int* q) {

}

__global__ void foo(int* p) {

  __shared__ int A[10];

  bar(p);

  bar(A);

}

