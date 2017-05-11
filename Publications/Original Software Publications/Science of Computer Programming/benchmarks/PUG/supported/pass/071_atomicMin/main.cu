#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, unsigned int* B, unsigned long long int* C)
{
  atomicMin(A,10);

  atomicMin(B,1);

  atomicMin(C,5);
}

#endif
