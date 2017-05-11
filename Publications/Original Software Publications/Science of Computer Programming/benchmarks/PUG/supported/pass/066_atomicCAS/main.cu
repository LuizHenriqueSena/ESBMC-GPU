#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, unsigned int* B, unsigned long long int* C)
{
  atomicCAS(A,2,11);

  atomicCAS(B,5,1);

  atomicCAS(C,7,3);
}

#endif
