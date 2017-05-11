/*** substitui os valores aleatórios de determinado vetor de tamanho N por valores ordenados de 0 a N ***/
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#define N 2//64

__global__ void foo(int* glob) {

  int a;

  int* p;

  a = 0;

  p = &a;

  *p = threadIdx.x;

  glob[*p] = threadIdx.x;
}

