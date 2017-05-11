#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

#define N 2//64

__device__ int bar () {
  return 0;
}

__global__ void foo() {
  __assert(bar () !=0);
}

