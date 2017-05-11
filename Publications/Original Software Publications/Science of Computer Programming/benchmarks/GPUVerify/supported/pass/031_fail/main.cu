/******************************************* ternarytest2.cu ***************************************/
/*mostra 0 no índice 0, 2.4 no índice 1 e nos índice ímpares, mostra valor lixo nos demais índices */
//fail: assert

#include <stdio.h>
#include "cuda.h"

#include <assert.h>
#define N 2 //64

__global__ void foo(float* A, float c) {

  A[threadIdx.x ? 2*threadIdx.x : 1] = c;

}

