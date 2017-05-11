//fail: assertion
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#define N 2//1024

__global__ void foo(float *A, int sz) {
	__requires (sz == N)
	__assert(sz == blockDim.x);
	for(int i = threadIdx.x; i < 1*sz; i += sz) {
		A[i] *= 2.0f;
  }
}

