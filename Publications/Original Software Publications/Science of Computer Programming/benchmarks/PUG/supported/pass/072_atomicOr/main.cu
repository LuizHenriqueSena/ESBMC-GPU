#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, unsigned int* B, unsigned long long int* C)
{
  atomicOr(A,10);

  atomicOr(B,7);

  atomicOr(C,5);
}

#endif
