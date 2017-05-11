//pass
//--blockDim=64 --gridDim=64 --no-inline

#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#define N 2

__device__ void bar(int* q) {

}

__global__ void foo(int* p) {

  __shared__ int A[10];

  bar(p);

  bar(A);

}

int main(){

	int* a;

	foo<<<N,N>>>(a);
	//ESBMC_verify_kernel(foo,1,N,a);

	cudaThreadSynchronize();
}
