#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* A) {

   A[ blockIdx.x*blockDim.x + threadIdx.x ] += (A[ (blockIdx.x + 1)*blockDim.x + threadIdx.x ]);

}

#endif
