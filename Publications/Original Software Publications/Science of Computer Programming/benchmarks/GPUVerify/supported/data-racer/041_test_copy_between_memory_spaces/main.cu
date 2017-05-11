//pass
//--blockDim=64 --gridDim=1 --equality-abstraction --no-inline

#include "cuda.h"
#include <stdio.h>
#include <assert.h>
#define N 2

__global__ void foo(int* p) {

	__shared__  int A[10];

	A[0] = 1;

	p[0] = A[0];

}

