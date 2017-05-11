//xfail:BOOGIE_ERROR
//main.cu: error: possible read-write race
//however, this didn't happen in the tests
//altough in CUDA providing the inline keyword should still keep a copy of the function around,
//this kind of access is considered a error by ESBMC
//ps: the values from A[N-1-offset) to A[N-1] always will receive unpredictable values,
//because they acess values because they access memory positions that were not initiated

#include <stdio.h>
#include <cuda.h>

#define tid threadIdx.x
#define N 2//1024

__device__ inline void inlined(int *A, int offset)
{
   int temp = A[tid + offset];
   A[tid] += temp;
}

__global__ void inline_test(int *A, int offset) {
  inlined(A, offset);
}

