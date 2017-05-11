#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int *c, const int *a){
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
	int row = (blockDim.y * blockIdx.y) + threadIdx.y;
    c[index(row,col,4)] = a[index(col, row, 4)] ;
}

#endif
