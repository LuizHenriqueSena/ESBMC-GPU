//xfail:BOOGIE_ERROR
//--gridDim=1 --blockDim=4 --no-inline
//attempt to modify constant memory

#include <stdio.h>
#include <cuda.h>


#define N 2//4

__constant__ int global_constant[N]; //= {0, 1, 2, 3};

__global__ void foo(int *in) {

	global_constant[threadIdx.x] = in[threadIdx.x];

	__syncthreads();

	in[threadIdx.x] = global_constant[threadIdx.x];

}

