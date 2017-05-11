#include <cuda.h>


#include <stdio.h>
#include <assert.h>

#define N 2//8

__device__  double C[2][2][2];

__device__ int index (int a, int b, int c){
	return 4*a + 2*b + c;
}

__global__ void foo(double *H) {

	int idx = index (threadIdx.x,threadIdx.y,threadIdx.z);

	H[idx] = C[threadIdx.x][threadIdx.y][threadIdx.z];
}

