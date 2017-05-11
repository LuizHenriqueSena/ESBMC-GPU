//pass
//--blockDim=256 --gridDim=1 --no-inline

#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>
//#include <curand_precalc.h>
//#include <curand_mtgp32_host.h>
#include <stdio.h>

#define N 256

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {

	A[threadIdx.x] = curand(&state[threadIdx.x]);
}

