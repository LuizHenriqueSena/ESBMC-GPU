#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(curandState *state, float *A) { // test: replace curandState for curandStateXORWOW_t
   A[threadIdx.x] = curand_uniform(state);
}

#endif
