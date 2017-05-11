//pass
//--blockDim=64 --gridDim=64 --no-inline

#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#define N 2//64

__device__ void bar(float x) {
	__assert(1);
}

__global__ void foo(int* A) {

  bar(A[0]);

}

