//fail: o argumento não é passado com sucesso
//--blockDim=1024 --gridDim=1 --no-inline
#include <stdio.h>
#include <stdlib.h>

#define N 8

__device__ float multiplyByTwo(float *v, unsigned int tid) {

    return v[tid] * 2.0f;
}

__device__ float divideByTwo(float *v, unsigned int tid) {

    return v[tid] * 0.5f;
}

typedef float(*funcType)(float*, unsigned int);

__global__ void foo(float *v, funcType* f, unsigned int size)
{
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__assert(*f == divideByTwo || *f == multiplybyTwo);

    if (tid < size) {
        v[tid] = (*f)(v, tid);
    }
}

