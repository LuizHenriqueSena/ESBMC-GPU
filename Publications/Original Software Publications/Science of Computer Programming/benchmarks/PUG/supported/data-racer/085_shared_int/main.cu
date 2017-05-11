#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel() {

  __shared__ int a;

  a = threadIdx.x;
}

#endif
