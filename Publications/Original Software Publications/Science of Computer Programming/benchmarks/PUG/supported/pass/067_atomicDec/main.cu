#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (unsigned int* B)
{
  atomicDec(B,7);//0111 -> 1000 -> 0000 -> 0001 -> 0010 -> 0011 -> 0100 -> 0101 -> 0110 ...
  	  /*the second argument on atomicDec() is a limit for decs. When this limit is reached, B receives <LIM>*/
}

#endif
