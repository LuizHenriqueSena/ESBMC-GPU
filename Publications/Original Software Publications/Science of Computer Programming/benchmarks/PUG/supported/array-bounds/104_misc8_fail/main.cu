#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(char *A, char *B, char* c)
{

  char *choice1 = *c ? A : B;	//It Makes choice1 receives A
  char *choice2 = *c ? B : A;	//It Makes choice2 receives B

  bar(&choice1, &choice2);
  bar(&choice1, &choice2);
}

#endif
