#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <assert.h>

#define N 2//64

__device__ int bar(float* A) {

  if(threadIdx.x != 0) {
	return 0;
  }

 return 1;

}

__global__ void foo(float* A) {

  int y = bar(A);
  A[threadIdx.x]=y;

}

