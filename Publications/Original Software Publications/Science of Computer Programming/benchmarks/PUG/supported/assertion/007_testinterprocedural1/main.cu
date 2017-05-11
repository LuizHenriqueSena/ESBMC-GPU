#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel (int* p, int* q){

    bar(p);

    bar(q);
	assert(*p == 2);
    //*p = 23; *q = 23; // remove this comment to see that the __device__ function does not work
}

#endif
