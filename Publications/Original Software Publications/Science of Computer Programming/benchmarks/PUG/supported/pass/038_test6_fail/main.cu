#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* p) {

  //bar(p)[threadIdx.x] = 0;
  *(bar(p)+threadIdx.x) = 2;
  //printf(" %d; ", bar(p)[threadIdx.x]);

}

#endif
