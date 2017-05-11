//fail
//--blockDim=256 --gridDim=1 --no-inline

#include <curand_kernel.h>
//#include <curand_mtgp32_host.h>
#include <stdio.h>

#define N 32 //16

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {
  if (threadIdx.x == 0) {
    A[blockIdx.x] = curand(state);
  }
}

