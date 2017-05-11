#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <assert.h>

#define N 2//64

__device__ int f(int x) {
	
	return x + 2;
}

__global__ void foo(int *y, int x) {

	*y = f(x);

}

