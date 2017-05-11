#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* p) {

  int* q = bar(p);

  q[threadIdx.x] = 0;
  //printf(" %d; ", q[threadIdx.x]);

}

#endif
