#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float *v, funcType f, funcType g, unsigned int size)
{
	assert(f == divideByTwo);
	assert(g == multiplyByTwo);

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    funcType h;
    
    if (f >= g)
      h = f;
    else
      h = g;

    if (tid < size)
    {
        v[tid] = (*h)(v, tid);
    }
}

#endif

