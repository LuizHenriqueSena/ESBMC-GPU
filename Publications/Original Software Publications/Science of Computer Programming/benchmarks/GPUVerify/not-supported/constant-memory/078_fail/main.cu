//xfail:BOOGIE_ERROR
//--blockDim=8 --gridDim=1 --no-inline

// The statically given values for A are not preserved when we translate CUDA
// since the host is free to change the contents of A.
// cf. testsuite/OpenCL/globalarray/pass2

#include <stdio.h>
#include <assert.h>
#include <cuda.h>


#define N 2//8
#define THREAD_CHANGE 1


__constant__ float A[8] = {0,1,2,3,4,5,6,7};

__global__ void globalarray(float* p) {
  int i = threadIdx.x;
  A[THREAD_CHANGE] = 0;		// forçando a entrada no laço, alterando uma constante!
  int a = A[i];

  if(a != threadIdx.x) {
    p[0] = threadIdx.x;	  //entra aqui apenas para para thread=1, por isso não há corrida de dados
  }
}

