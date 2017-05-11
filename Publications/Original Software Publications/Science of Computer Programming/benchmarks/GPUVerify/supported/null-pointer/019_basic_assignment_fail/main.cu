//xfail:BOOGIE_ERROR
//error: possible null pointer access

#include <stdio.h>
#include <cuda.h>

#define N 2//64

__device__ float multiplyByTwo(float *v, unsigned int tid)
{
    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid)
{
    return v[tid] * 0.5f;
}

typedef float(*funcType)(float*, unsigned int);

__global__ void foor(float *v, unsigned int size, unsigned int i)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    funcType f;

	__requires (i = 1 || i = 2)
	/*** ESBMC_assert(i =1 || i = 2, "possible null pointer access"); ***/
    if (i == 1)
      f = multiplyByTwo;
    else if (i == 2)
      f = divideByTwo;
    else
      f = NULL;

    if (tid < size)
    {
        float x = (*f)(v, tid);
        x += multiplyByTwo(v, tid);
        v[tid] = x;
    }
}

