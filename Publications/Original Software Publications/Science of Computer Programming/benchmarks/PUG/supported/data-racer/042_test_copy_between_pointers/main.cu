#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* p) {

  __shared__ int A[10];

  int* x;

  x = p;

  	assert(*p <2);

  x[0] = 0;
	
  x = A;

  x[0] = 0;

}

#endif
