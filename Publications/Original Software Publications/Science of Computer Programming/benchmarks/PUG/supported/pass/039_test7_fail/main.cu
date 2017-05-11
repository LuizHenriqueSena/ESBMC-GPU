#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* glob) {

  int a;

  int* p;

  a = 0;

  p = &a;

  *p = threadIdx.x;

  glob[*p] = threadIdx.x;
}

#endif
