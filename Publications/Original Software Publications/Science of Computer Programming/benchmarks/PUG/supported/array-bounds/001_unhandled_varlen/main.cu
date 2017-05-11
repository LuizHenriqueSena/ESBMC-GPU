#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(uint4 *out) {
  uint4 vector;
  int len = bar();
   memset(&vector, 5, len); /*modify manually the value of len to see the bugs*/
  out[threadIdx.x] = vector;
}

#endif
