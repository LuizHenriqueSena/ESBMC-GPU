//pass
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

#define N 2//32

__global__ void kernel(uint4 *out) {
  uint4 vector = {1,1,1,1};
  out[threadIdx.x] = vector;
}

