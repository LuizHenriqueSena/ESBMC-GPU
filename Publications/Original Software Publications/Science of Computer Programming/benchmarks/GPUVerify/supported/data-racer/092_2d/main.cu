//data-racer

#include <cuda.h>

#include <stdio.h>

#define SIZE 2
#define TILES 4
#define LENGTH (TILES * SIZE)
#define N 2

__global__ void matrix_transpose(float* A)
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

