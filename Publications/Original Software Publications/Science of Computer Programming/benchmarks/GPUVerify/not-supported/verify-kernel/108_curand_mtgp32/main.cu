//pass
//--blockDim=256 --gridDim=1 --no-inline

#include <cuda.h>
#include <curand_kernel.h>
#include <curand_mtgp32.h>
#include <stdio.h>
//#include <curand.h>

#define N 2 //256

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {

	A[threadIdx.x] = curand(&state[threadIdx.x]);
}

