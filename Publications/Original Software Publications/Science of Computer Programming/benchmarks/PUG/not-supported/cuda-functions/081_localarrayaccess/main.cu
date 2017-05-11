#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel() {

  __shared__ int A[10];

  A[threadIdx.x] = 2;

  __syncthreads (); //evita corrida de dados

  int x = A[threadIdx.x + 1];

}

#endif
