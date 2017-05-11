//pass
//--blockDim=1024 --gridDim=1 --boogie-file=${KERNEL_DIR}/axioms.bpl --no-inline
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#define N 8

typedef float(*funcType)(float*, unsigned int);

__device__ float multiplyByTwo(float *v, unsigned int tid)
{
    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid)
{
    return v[tid] * 0.5f;
}

// Static pointers to device functions

	__device__ funcType p_mul_func = multiplyByTwo;

	__device__ funcType p_div_func = divideByTwo;


__global__ void foo(float *v, funcType f, unsigned int size, int i)
{
    __assert(f == divideByTwo);
	__assert(i != 0);

	uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    void *x = (void*)f; /*ptr_to_ptr*/

    if (i == 0)
      x = NULL;

    funcType g = (funcType)x;

    if (tid < size)
    {
        v[tid] = (*g)(v, tid);
    }
}
