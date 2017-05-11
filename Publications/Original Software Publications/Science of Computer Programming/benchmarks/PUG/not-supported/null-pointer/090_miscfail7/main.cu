#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int *H) {
  size_t tmp = (size_t)H; //type cast
  tmp += sizeof(int);
  int *G = (int *)tmp;
  G -= 1;					//POSSIBLE NULL POINTER ACCESS
  G[threadIdx.x] = threadIdx.x;
  __syncthreads();
  H[threadIdx.x] = G[threadIdx.x];
}

#endif
