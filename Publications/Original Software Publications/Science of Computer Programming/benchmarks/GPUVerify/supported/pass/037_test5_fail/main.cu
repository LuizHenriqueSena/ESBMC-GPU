#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#define N 2//64

__device__ int* bar(int* p) {
  return p;
}

__global__ void foo(int* p) {

  int* q = bar(p);

  q[threadIdx.x] = 0;
  //printf(" %d; ", q[threadIdx.x]);

}

