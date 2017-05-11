#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(uint4 *out) {
  uint4 vector = {1,1,1,1};
  out[threadIdx.x] = vector;
}

#endif
