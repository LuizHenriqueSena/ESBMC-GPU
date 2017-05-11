#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

struct S {
  int * p;
};

__global__ void* kernel(int * A) {

  S myS;
  myS.p = A;
  int * q;
  q = myS.p;
  q[threadIdx.x + blockDim.x*blockIdx.x] = threadIdx.x;

}

#endif
