//pass
//--blockDim=1024 --gridDim=1 --no-inline
#include <stdio.h>
#include <stdlib.h>
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

__global__ void foog(float *v, funcType f, funcType g, unsigned int size)
{
	__assert(f == divideByTwo);
	__assert(g == multiplyByTwo);

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    funcType h;
    
    if (f >= g)
      h = f;
    else
      h = g;

    if (tid < size)
    {
        v[tid] = (*h)(v, tid);
    }
}

