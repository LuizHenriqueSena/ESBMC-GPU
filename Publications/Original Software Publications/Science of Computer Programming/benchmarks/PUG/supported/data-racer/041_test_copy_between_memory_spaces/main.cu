#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int* p) {

	__shared__  int A[10];

	A[0] = 1;

	p[0] = A[0];

}

#endif
