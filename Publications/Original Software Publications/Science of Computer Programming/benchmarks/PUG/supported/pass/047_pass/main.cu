#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int *c) {
  int b, a;
  a = 2;
  b = 3;
  c[threadIdx.x]= a+b;
  __syncthreads ();
}

#endif
