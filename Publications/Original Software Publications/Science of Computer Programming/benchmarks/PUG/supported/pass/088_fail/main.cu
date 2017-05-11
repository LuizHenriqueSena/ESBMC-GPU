#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, int* B)
{
  int idxa          = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int idxb = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

  A[idxa] = idxa;
  B[idxb] = idxa;
}

#endif
