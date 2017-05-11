// data-racer
#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#define N 2

__global__ void foo(int	*p, int *ptr_a) {

	ptr_a = p + threadIdx.x;

//	assert(*ptr_a == 2);
}

