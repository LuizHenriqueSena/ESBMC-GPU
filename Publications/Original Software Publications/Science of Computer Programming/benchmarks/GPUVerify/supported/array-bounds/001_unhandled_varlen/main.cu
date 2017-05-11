//xfail:BUGLE_ERROR
//--gridDim=1 --blockDim=32 --no-inline

//This kernel is racy: memset is called with variable length.
//#define memset(dst,val,len) __builtin_memset(dst,val,len)

#define N 2//32

#include <stdio.h>
#include <cuda.h>


__device__ int bar(void){
	int value;
	return value;
}

__global__ void kernel(uint4 *out) {
  uint4 vector;
  int len = bar();
   memset(&vector, 5, len); /*modify manually the value of len to see the bugs*/
  out[threadIdx.x] = vector;
}
