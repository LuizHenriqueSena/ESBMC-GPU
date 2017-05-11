#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

typedef struct {
  float x,y,z,w;
} myfloat4;

__global__ void kernel(float * i0) {
  myfloat4 f4;
  f4.x = 2;
  i0[threadIdx.x + blockDim.x*blockIdx.x] = f4.x;
}

#endif
