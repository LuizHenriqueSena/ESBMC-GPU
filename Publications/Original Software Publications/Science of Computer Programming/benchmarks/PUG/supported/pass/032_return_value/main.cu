#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float *v, unsigned int size, int i)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    funcType f = grabFunction(i);

    if (tid < size)
    {
    	float x = (*f)(v, tid);
		x += multiplyByTwo(v, tid);
		v[threadIdx.x] = x;
    }
}

#endif
