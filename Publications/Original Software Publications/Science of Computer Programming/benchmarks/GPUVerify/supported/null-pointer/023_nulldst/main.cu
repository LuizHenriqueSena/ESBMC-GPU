//xfail:BOOGIE_ERROR
//--gridDim=1 --blockDim=32 --no-inline

//#define memset(dst,val,len) __builtin_memset(dst,val,len)

#define N 2//32

#include <stdio.h>
#include <cuda.h>


__global__ void kernel(uint4 *out) {
  uint4 vector;
  memset(0, 0, 16);
  out[threadIdx.x] = vector;
  /**/
}

