//TEST CASE PASS IN GPU_VERIFY. IT IS NOT VERIFY ARRAY BOUNDS VIOLATION

#include <stdio.h>
#include <cuda.h>

#include <assert.h>


#define N 2//64

__global__ void foo(int* p) {
  int* q;
  q = p + 1;
  p[threadIdx.x] = q[threadIdx.x];
}

