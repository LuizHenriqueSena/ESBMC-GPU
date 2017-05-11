//fail
//--blockDim=64 --gridDim=64 --no-inline

#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#define DIM  2//64
#define N 2 //DIM*DIM

typedef struct {
  float x,y,z,w;
} myfloat4;

__global__ void k(float * i0) {
  myfloat4 f4;
  f4.x = 2;
  i0[threadIdx.x + blockDim.x*blockIdx.x] = f4.x;
}

