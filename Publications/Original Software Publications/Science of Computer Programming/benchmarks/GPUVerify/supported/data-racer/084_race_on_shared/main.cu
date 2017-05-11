#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>


#define N 2

__global__ void foo()
{
	__shared__ int A[8];
	A[0] = threadIdx.x;
}

