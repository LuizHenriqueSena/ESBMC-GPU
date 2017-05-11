//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1
//null pointer access
// ALTOUGH, IT WORKS

#include <stdio.h>
#include <cuda.h>
 
#define N 2//4//8

__global__ void foo(int *H) {
  size_t tmp = (size_t)H; //type cast
  tmp += sizeof(int);
  int *G = (int *)tmp;
  G -= 1;					//POSSIBLE NULL POINTER ACCESS
  G[threadIdx.x] = threadIdx.x;
  __syncthreads();
  H[threadIdx.x] = G[threadIdx.x];
}

