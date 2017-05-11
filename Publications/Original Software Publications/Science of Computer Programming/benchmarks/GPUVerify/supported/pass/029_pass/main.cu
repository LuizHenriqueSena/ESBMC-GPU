#include <cuda.h>
#include <assert.h>
#define N 2//(64*64)//(2048*2048)
#define THREADS_PER_BLOCK 2//512

__global__ void Asum(int *a, int *b, int *c){
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	c[index] = a[index] + b[index];
}

