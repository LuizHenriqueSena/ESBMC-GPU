//xfail:BOOGIE_ERROR
//--blockDim=32 --gridDim=64 --no-inline
//error: possible write-write race on

#include <stdio.h>
#include "cuda.h"
#include <assert.h>


#define M 2//32
#define N 4//64

__global__ void foo(int* p) {
  __shared__ unsigned char x[N];

  for (unsigned int i=0; i<(N/4); i++) {
    ((unsigned int*)x)[i] = 1;//0;
  }
/*
  for (int i = 0; i < N/4; i++) {
	  p[i] = x[i];
  }
*/
}

