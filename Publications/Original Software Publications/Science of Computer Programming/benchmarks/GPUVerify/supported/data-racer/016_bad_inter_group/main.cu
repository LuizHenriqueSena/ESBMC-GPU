//xfail:BOOGIE_ERROR
//--blockDim=128 --gridDim=128 --warp-sync=32 --no-inline
//kernel.cu: error: possible read-write race on A
//It fail to dim >= 128, because it can't synchronize.
#include <stdio.h>
#include <cuda.h>

#define N dim*dim
#define dim 2//128 //64

__global__ void foo(int* A) {

   A[ blockIdx.x*blockDim.x + threadIdx.x ] += (A[ (blockIdx.x + 1)*blockDim.x + threadIdx.x ]);

}

