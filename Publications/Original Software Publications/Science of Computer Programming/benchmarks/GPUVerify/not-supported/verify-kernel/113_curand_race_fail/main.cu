//xfail:BOOGIE_ERROR
//--blockDim=2 --gridDim=1 --no-inline
//Write by thread .+kernel\.cu:8:21:

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <assert.h>

#define N 4

__global__ void curand_test(curandState *state, float *A) { // test: replace curandState for curandStateXORWOW_t
   A[threadIdx.x] = curand_uniform(state);
}

