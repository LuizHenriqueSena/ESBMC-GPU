#include <cuda.h>
//#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <stdio.h>
#include <assert.h>

#define N 16

__device__ int index(int col, int row, int ord){
	return (row *ord)+col;
}

__global__ void Transpose(int *c, const int *a){
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
	int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    c[index(row,col,4)] = a[index(col, row, 4)] ;
}

