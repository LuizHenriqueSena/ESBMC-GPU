#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

#include <assert.h>

__global__ void Asum(int *a, int *b, int *c){
	*c = *a + *b;
}

