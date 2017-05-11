//pass
//--blockDim=64 --gridDim=64 --equality-abstraction --no-inline

#include "cuda.h"
#include <stdio.h>
#include <assert.h>

#define N 2

__global__ void foo(int* p) {

	__requires(*p == 1)
  __shared__ int A[10];

  int* x;
	
  x = p;
	
  	__assert(*p <2);

  x[0] = 0;
	
  x = A;

  x[0] = 0;

}

