//xfail:data-race
// Write by thread 0
// Write by thread 1
// x = 1

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 2//512

__global__ void example(float * A, int x) {

	__requires(x = 1); // x deve ser a diferença entre o limite do if1 e do if2

    if(threadIdx.x == 0) {
        A[threadIdx.x + x] = threadIdx.x; //A[1] = 0;
    }

    if(threadIdx.x == 1) {
        A[threadIdx.x] = threadIdx.x; //A[1] = 1;
   }
}

