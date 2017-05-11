//pass
//--blockDim=1024 --gridDim=1 --no-inline

#include <cuda.h>
#include <stdio.h>

#define N 2 //1024

__global__ void definitions (int* A, unsigned int* B, unsigned long long int* C)
{
  atomicXor(A,10);//1010 xor 0101 = 1111 /*xor looks for distinct bits*/

  atomicXor(B,7);//0111 xor 0101 = 0010

  atomicXor(C,5);//0101 xor 0101 = 0000
}

