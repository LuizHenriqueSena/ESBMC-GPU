//pass
//--blockDim=64 --gridDim=64 --no-inline
#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define N 2//64


__global__ void foo() {

  float x = (float)2;

}

