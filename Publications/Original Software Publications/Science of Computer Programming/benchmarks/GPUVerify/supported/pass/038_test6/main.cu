#include <stdio.h>
#include <cuda.h>
#include <assert.h>

#define N 2//64

__device__ int* bar(int* p) {

	//__ensures(__implies(__enabled(), __return_val_ptr() == p));
	return p;
}

__global__ void foo(int* p) {

  //bar(p)[threadIdx.x] = 0;
  *(bar(p)+threadIdx.x) = 2;
  //printf(" %d; ", bar(p)[threadIdx.x]);

}

