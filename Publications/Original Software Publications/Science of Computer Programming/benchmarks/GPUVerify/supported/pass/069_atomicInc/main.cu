//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda.h>
#include <stdio.h>

#define N 2 //1024

__global__ void definitions (unsigned int* B)
{
  atomicInc(B,7);//0111 -> 1000 -> 0000 -> 0001 -> 0010 -> 0011 -> 0100 -> 0101 -> 0110 ...
  	  /*the second argument on atomicInc() is a limit for increments. When this limit is reached, B receives 0*/
}

