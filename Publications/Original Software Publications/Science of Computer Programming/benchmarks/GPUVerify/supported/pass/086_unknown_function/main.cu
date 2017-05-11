//pass
//blockDim=1024 --gridDim=1 --no-inline

#include <stdio.h>
#include <cuda.h>

#include <math_functions.h>

typedef double(*funcType)(double);

__device__ double bar(double x) {
  return sin(x);
}

__global__ void foo(double x, int i)
{
	__requires(i==1);
  funcType f;

  if (i == 0)
    f = bar;
  else
    f = cos;

  double z = f(x);
	__assert(z != NULL);

  printf("z: %f ", z);
}

