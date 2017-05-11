//xfail:BOOGIE_ERROR
//--blockDim=2 --gridDim=1 --no-inline
//Write by thread .+kernel.cu:8:4:
// to threadIdx.x != 0 we have 'data race'.

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#define N 8 //2

__global__ void init_test(curandState *state, unsigned int *A) {
   curand_init(0, 0, 0, state);

   __syncthreads();

   A[threadIdx.x] =  curand(&state[threadIdx.x]);
//   if (threadIdx.x == 0) {
  //   A[0] = curand(state);
   //}
}

