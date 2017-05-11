#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(volatile int* p)
{
    //__assert(__no_read(p));
    p[threadIdx.x] = threadIdx.x;
}

#endif
