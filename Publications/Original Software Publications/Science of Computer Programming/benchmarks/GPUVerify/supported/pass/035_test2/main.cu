#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#define N 2//64

__global__ void foo(int* p) {

  int* q;

  q = p;

  q[threadIdx.x] = 0;

}

