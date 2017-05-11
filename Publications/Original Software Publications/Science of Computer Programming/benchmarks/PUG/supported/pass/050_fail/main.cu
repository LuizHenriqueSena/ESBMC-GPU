#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* x) {
  x[threadIdx.x] = A[threadIdx.x] + B[0]; //permanece constante por ser muito grande. N < 1024 não permanece
}

#endif
