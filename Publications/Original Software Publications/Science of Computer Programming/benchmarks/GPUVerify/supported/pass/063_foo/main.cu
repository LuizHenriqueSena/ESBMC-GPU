#include <stdio.h>
#include "cuda.h"
#include <assert.h>
#define N 2 //16

__device__ int bar(int x) {

	return x + 1;
}

__global__ void foo(int *A) {

	A[threadIdx.x] = bar(threadIdx.x);
}

