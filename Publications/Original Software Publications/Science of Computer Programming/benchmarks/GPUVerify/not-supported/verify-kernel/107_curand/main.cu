//pass
//--blockDim=512 --gridDim=1 --no-inline

#include <cuda.h>
#include <stdio.h>
#include <assert.h>


#define N 2 //512

__global__ void curand_test(curandState *state, float *A) {
   A[threadIdx.x] =  curand(&state[threadIdx.x]); // the pseudo random number returned by 'curand' is an unsigned int
}

