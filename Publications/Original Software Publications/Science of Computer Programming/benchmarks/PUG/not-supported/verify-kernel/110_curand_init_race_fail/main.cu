#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(curandState *state, unsigned int *A) {
   curand_init(0, 0, 0, state);

   __syncthreads();

   A[threadIdx.x] =  curand(&state[threadIdx.x]);
//   if (threadIdx.x == 0) {
  //   A[0] = curand(state);
   //}
}

#endif
