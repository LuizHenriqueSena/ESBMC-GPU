#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float *v, funcType f, unsigned int size, int i)
{
    assert(f == divideByTwo);
	assert(i != 0);

	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    void *x = (void*)f; /*ptr_to_ptr*/

    if (i == 0)
      x = NULL;

    funcType g = (funcType)x;

    if (tid < size)
    {
        v[tid] = (*g)(v, tid);
    }
}

#endif
