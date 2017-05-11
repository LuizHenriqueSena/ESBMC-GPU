#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel()
{
  int x = bar();
  assert(x == 5);
//  printf("%d ", x);

}

#endif
