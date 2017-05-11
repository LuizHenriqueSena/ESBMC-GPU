//xfail:BOOGIE_ERROR
//--blockDim=64 --gridDim=64 --no-inline
//
#include "cuda.h"
#define N dim*dim
#define dim 2

__global__ void foo() {

  __shared__ int a;

  a = threadIdx.x;
}

