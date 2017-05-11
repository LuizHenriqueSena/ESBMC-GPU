#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float* A)
{
  __shared__ float tile [SIZE][SIZE];

  int x = threadIdx.x;
  int y = threadIdx.y;

  int tile_x = blockIdx.x;
  int tile_y = blockIdx.y;

	tile[x][y] = A[((x + (tile_x * SIZE)) * LENGTH) + (y + (tile_y * SIZE))];

	tile[x][y] = tile[y][x];

	__syncthreads();

	A[((x + (tile_y * SIZE)) * LENGTH) + (y + (tile_x * SIZE))] = tile[x][y];
}

#endif
