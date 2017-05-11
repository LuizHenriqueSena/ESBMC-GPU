/******************************************* ternarytest.cu **************************************/
/*mostra 0 no índice 0, "c" no índice 1 e nos índice pares, mostra valor lixo nos demais índices */

#include <stdio.h>
#include "cuda.h"

#include <assert.h>

#define N 2//64

__global__ void foo(float* A, float c) {

		A[threadIdx.x == 0 ? 1 : 2*threadIdx.x] = c;

}

