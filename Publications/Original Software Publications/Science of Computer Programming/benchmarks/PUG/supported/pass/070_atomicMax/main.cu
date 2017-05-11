#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, unsigned int* B, unsigned long long int* C)
{
  atomicMax(A,10);

  atomicMax(B,1);

  atomicMax(C,5);
}

#endif
