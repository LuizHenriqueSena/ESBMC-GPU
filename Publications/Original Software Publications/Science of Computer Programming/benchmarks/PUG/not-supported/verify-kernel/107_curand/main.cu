#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(curandState *state, float *A) {
   A[threadIdx.x] =  curand(&state[threadIdx.x]); // the pseudo random number returned by 'curand' is an unsigned int
}

#endif
