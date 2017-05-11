//pass: checka a função device (comparar com o cuda69_test2)
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#define N 2//64

__device__ void bar(int* p) {
  p[threadIdx.x] = 0;
  //printf(" %d; ", p[threadIdx.x]);
}

__global__ void foo(int* p) {

  bar(p);

}

