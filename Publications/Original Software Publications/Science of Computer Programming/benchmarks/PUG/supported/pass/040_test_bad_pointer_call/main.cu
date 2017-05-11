#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* p) {

  __shared__ int A[10];

  bar(p);

  bar(A);

}

#endif
