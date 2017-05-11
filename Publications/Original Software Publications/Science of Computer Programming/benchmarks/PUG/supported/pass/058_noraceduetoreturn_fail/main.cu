#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float* A) {

  int y = bar(A);
  A[threadIdx.x]=y;

}

#endif
