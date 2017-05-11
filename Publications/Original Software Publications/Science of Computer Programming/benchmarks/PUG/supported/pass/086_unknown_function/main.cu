#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(double x, int i)
{
  funcType f;

  if (i == 0)
    f = bar;
  else
    f = cos;

  double z = f(x);
	assert(z != NULL);

  printf("z: %f ", z);
}

#endif
