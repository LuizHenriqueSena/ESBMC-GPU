#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* p) {
  __shared__ unsigned char x[N];

  for (unsigned int i=0; i<(N/4); i++) {
    ((unsigned int*)x)[i] = 1;//0;
  }
/*
  for (int i = 0; i < N/4; i++) {
	  p[i] = x[i];
  }
*/
}

#endif
