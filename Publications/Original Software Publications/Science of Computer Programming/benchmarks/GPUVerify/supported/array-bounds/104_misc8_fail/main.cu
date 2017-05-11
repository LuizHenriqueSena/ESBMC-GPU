//fail
//--blockDim=1024 --gridDim=1

#include <cuda.h>

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define N 2//1024

//replace out with in
__device__ void bar(char **in, char **out) {
	char tmp = (*in)[threadIdx.x];
	  out[0][threadIdx.x] = tmp;
	  *out = *in;
}

__global__ void foo(char *A, char *B, char* c)
{

	__requires(c = A);

  char *choice1 = *c ? A : B;	//It Makes choice1 receives A
  char *choice2 = *c ? B : A;	//It Makes choice2 receives B

  bar(&choice1, &choice2);
  bar(&choice1, &choice2);
}

