#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* p) {
  int* q;
  q = p + 1;
  p[threadIdx.x] = q[threadIdx.x];
}

#endif
