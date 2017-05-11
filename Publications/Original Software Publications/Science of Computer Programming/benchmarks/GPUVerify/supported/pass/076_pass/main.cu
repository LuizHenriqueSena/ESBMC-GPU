#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <assert.h>

#define N 2//64

__global__ void foo(float* A) {

	__requires(*A == 0);
  if(blockIdx.x == 0)
	  A[threadIdx.x] = 50.f;
}

