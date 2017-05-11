#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(unsigned a, unsigned b, unsigned c) {

    //__assert(a + b != c);
	assert((a+b) != c);

}

#endif
