#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <assert.h>
//#include <time.h>
#define N 2	//512

__global__ void Asum(int *a, int *b, int *c){
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

