#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, unsigned int* B, unsigned long long int* C)
{
  atomicAnd(A,10);

  atomicAnd(B,1);

  atomicAnd(C,5);
}

#endif
