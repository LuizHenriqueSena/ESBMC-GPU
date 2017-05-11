#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float *v)
{
    funcType f = (funcType)3; // it's a null pointer
    v[tid] = (*f)(v, tid);
}

#endif
