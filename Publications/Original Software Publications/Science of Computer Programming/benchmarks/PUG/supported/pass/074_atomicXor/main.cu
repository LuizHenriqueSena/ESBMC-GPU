#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* A, unsigned int* B, unsigned long long int* C)
{
  atomicXor(A,10);//1010 xor 0101 = 1111 /*xor looks for distinct bits*/

  atomicXor(B,7);//0111 xor 0101 = 0010

  atomicXor(C,5);//0101 xor 0101 = 0000
}

#endif
