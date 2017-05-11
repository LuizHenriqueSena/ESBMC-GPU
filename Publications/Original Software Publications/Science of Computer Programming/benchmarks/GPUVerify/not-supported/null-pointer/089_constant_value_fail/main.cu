//xfail:BOOGIE_ERROR
//--blockDim=1024 --gridDim=1 --no-inline
//error: possible null pointer access

#include <stdio.h>
#include <assert.h>
#include <cuda.h>


#define N 2//8

#define tid (blockIdx.x * blockDim.x + threadIdx.x)

__device__ float multiplyByTwo(float *v, unsigned int index)
{
    return v[index] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int index)
{
    return v[index] * 0.5f;
}

typedef float(*funcType)(float*, unsigned int);

__global__ void foo(float *v)
{
    funcType f = (funcType)3; // it's a null pointer
    v[tid] = (*f)(v, tid);
}

