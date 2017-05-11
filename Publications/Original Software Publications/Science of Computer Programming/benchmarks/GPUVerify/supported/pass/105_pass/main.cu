//pass
//--blockDim=1024 --gridDim=1
#include <cuda.h>


#include <stdio.h>
#include <string.h>
#include <assert.h>

#define N 4//1024

//swap the strings
__device__ void swap(char *in, char *out) {
	char tmp[N];
	tmp[threadIdx.x]= in[threadIdx.x];
	__syncthreads();
	in[threadIdx.x] = out[threadIdx.x];
	__syncthreads();
	out[threadIdx.x]= tmp[threadIdx.x];
}

__global__ void foo(char *A, char *B, char* c)
{
  char *choice1 = A;	//It Makes choice1 receives A
  char *choice2 = B;	//It Makes choice2 receives B
  swap(choice1, choice2);		//This function swaps choice1 and choice2
	__assert(strcmp(choice1,choice2) == 0);
}

