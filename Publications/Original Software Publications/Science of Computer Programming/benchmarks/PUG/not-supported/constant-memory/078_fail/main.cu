#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__constant__ float A[8] = {0,1,2,3,4,5,6,7};

__global__ void kernel(float* p) {
  int i = threadIdx.x;
  A[THREAD_CHANGE] = 0;		// forçando a entrada no laço, alterando uma constante!
  int a = A[i];

  if(a != threadIdx.x) {
    p[0] = threadIdx.x;	  //entra aqui apenas para para thread=1, por isso não há corrida de dados
  }
}

#endif
