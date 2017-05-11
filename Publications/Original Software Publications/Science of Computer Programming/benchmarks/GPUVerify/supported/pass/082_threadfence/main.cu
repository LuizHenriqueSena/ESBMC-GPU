//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>

__global__ void foo() {
    __threadfence();
}

