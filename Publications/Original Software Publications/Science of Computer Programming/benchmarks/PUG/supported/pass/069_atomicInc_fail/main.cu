#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (unsigned int* B)
{
  atomicInc(B,7);//0111 -> 1000 -> 0000 -> 0001 -> 0010 -> 0011 -> 0100 -> 0101 -> 0110 ...
  	  /*the second argument on atomicInc() is a limit for increments. When this limit is reached, B receives 0*/
}

#endif
