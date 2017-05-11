#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(uint4 *out) {
  uint4 vector;
  memset(0, 0, 16);
  out[threadIdx.x] = vector;
  /**/
}

#endif
