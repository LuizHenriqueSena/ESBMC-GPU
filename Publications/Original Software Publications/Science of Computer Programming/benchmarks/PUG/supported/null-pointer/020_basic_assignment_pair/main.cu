#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "MY_cutil.h"

__global__ void kernel(float *v, unsigned int size, unsigned int i) {

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    funcType f;
	assert(i == 1 || i == 2);
		
    if (i == 1)
      f = multiplyByTwo;
    else if (i == 2)
      f = divideByTwo;
    else
      f = NULL;

    if (tid < size)
    {
        float x = (*f)(v, tid);
        x += multiplyByTwo(v, tid);
		v[tid] = x;
    }
}

#endif
