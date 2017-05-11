//fail: data race
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#include <assert.h>

#define N 2//64

__global__ void foo (int* p, int* q){

    p[2] = q[2] + 1;

}

