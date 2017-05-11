#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__device__ void bar(int i);

__global__ void kernel(int w, int h)
{
   for (int i = threadIdx.x; //__invariant(h == 0);
	   i < w; i += blockDim.x)   {

		   if (h == 0)
			   bar(5);	
		   else
			   assert(0);
   }
}

#endif
