#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <assert.h>

#define N 2//64

__global__ void foo(int *c) {
  int b, a;
  a = 2;
  b = 3;
  c[threadIdx.x]= a+b;
  __syncthreads ();
}

