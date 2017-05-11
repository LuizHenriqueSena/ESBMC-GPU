#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>

#include <math_functions.h>

#define N 2//64

__global__ void foo(float *x, float y) {
	x[threadIdx.x] = __exp10f(y);	// pow(10,y), in this  case pow(10,2) = 100
}

