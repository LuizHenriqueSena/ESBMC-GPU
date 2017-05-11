#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include <assert.h>
static const int WORK_SIZE = /*256*/ 2;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */


__device__ unsigned int bitreverse1(unsigned int number) {
	number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
	number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
	number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
	return number;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bitreverse(void *data) {
	unsigned int *idata = (unsigned int*) data;
	idata[threadIdx.x] = bitreverse1(idata[threadIdx.x]);
}

