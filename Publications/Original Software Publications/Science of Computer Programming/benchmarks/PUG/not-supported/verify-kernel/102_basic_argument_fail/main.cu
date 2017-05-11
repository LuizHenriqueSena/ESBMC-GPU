#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float *v, funcType* f, unsigned int size)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	assert(*f == divideByTwo || *f == multiplybyTwo);

    if (tid < size) {
        v[tid] = (*f)(v, tid);
    }
}

#endif
