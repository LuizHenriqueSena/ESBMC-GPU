#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(curandStateMtgp32_t *state, float *A) {
  if (threadIdx.x == 0) {
    A[blockIdx.x] = curand(state);
  }
}

#endif
