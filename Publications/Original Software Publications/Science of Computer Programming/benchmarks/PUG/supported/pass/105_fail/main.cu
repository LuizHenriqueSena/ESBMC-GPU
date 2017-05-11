#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(char *A, char *B, char* c)
{
  char *choice1 = A;	//It Makes choice1 receives A
  char *choice2 = B;	//It Makes choice2 receives B
  swap(choice1, choice2);		//This function swaps choice1 and choice2
	assert(strcmp(choice1,choice2) != 0);
}

#endif
