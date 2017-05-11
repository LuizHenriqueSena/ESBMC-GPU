//xfail:BOOGIE_ERROR
//--blockDim=128 --gridDim=16 --no-inline
//assert\(false\)

#include <stdio.h>
#include <assert.h>
#include "cuda.h"



typedef void(*funcType)(float*);

__device__ void a(float *v)
{
	printf ("funcA with p%f = %f", *v, *v);
}
__device__ void b(float *v)
{
	printf ("funcB with p%f = %f", *v, *v);
}

__device__ void c(float *v)
{
	printf ("funcC with p%f = %f", *v, *v);
}

__device__ void d(float *v)
{
	printf ("funcD with p%f = %f", *v, *v);
}

__device__ void e(float *v)
{
	printf ("funcE with p%f = %f", *v, *v);
}

__global__ void should_fail(float * __restrict p1, float * __restrict p2, float * __restrict p3, float * __restrict p4, float * __restrict p5, int x, int y)
{
	__requires(x == 4);
	__requires(y == 4);
	funcType fp = a;

    switch(x) {
    case 1:
        fp = &a;
        break;
    case 2:
        fp = &b;
        break;
    case 3:
        fp = &c;
        break;
    case 4:
        fp = &d;
        break;
    default:
        fp = &e;
        break;
    }

    switch(y) {
    case 1:
        fp(p1);
        break;
    case 2:
        fp(p2);
        break;
    case 3:
        fp(p3);
        break;
    case 4:
        fp(p4);
        break;
    default:
        fp(p5);
        break;
    }

   __assert(0);
}

