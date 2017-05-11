//fail: assertion
//--blockDim=64 --gridDim=64 --no-inline

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <assert.h>

#define N 2//64

__device__ int f(int x) {
	
	return x + 1;
}

__global__ void foo(int *y) {

	*y = f(2);

}

