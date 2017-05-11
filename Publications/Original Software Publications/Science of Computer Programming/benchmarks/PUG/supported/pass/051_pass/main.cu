#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel() {

  __shared__ int A[10][10];
  A[threadIdx.y][threadIdx.x] = 2;
  assert(A[threadIdx.y][threadIdx.x]==2);
}

#endif
