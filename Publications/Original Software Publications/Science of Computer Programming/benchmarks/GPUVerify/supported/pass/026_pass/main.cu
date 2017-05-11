#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <assert.h>
//#include <time.h>
#define N 2

__global__ void MoreSums(int *a, int *b, int *c){
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

