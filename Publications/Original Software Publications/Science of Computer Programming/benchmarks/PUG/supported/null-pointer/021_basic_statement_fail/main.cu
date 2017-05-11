#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float *v, unsigned int i)
{
    funcType f;

    if (i == 1)
      f = multiplyByTwo;
    else if (i == 2)
      f = divideByTwo;
    else
      f = NULL;

    (*f)(v, tid);
}

#endif

