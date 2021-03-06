#include <call_kernel.h>
//xfail:BUGLE_ERROR
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is racy. However, variable-length memcpys are not supported.
//Expect error at Bugle stage.

#include <stdio.h>

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)

#define N 2

typedef struct {
  short x;
  short y;
  char z;
} s_t; //< sizeof(s_t) == 6

__global__ void overstep(s_t *in, s_t *out, size_t len) {
  memcpy(&out[threadIdx.x], &in[threadIdx.x], len);
}

int main(){
	s_t *a;
	s_t *dev_a;
	s_t *c;
	s_t *dev_c;
	int size = N*sizeof(s_t);

	a = (s_t*)malloc(size);
	c = (s_t*)malloc(size);

	/* initialization of a (the in) */
	a[0].x = 5; a[0].y = 6; a[0].z = 'i';
	a[1].x = 5; a[1].y = 6; a[1].z = 'i';

	/* initialization of c (the out) */
	c[0].x = 2; c[0].y = 3; c[0].z = 'o';
	c[1].x = 2; c[1].y = 3; c[1].z = 'o';

	cudaMalloc((void**)&dev_a, size);
	cudaMalloc((void**)&dev_c, size);

	cudaMemcpy(dev_a,a,size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c,c,size, cudaMemcpyHostToDevice);

	printf("a:\n");
	for (int i = 0; i < N; i++)
		printf("a[%d].x : %d  \ta[%d].y : %d\ta[%d].z : %c\n", i, a[i].x, i, a[i].y, i, a[i].z);

	printf("c:\n");
	for (int i = 0; i < N; i++)
		printf("c[%d].x : %d  \tc[%d].y : %d\tc[%d].z : %c\n", i, c[i].x, i, c[i].y, i, c[i].z);

	overstep<<<1,N>>>(dev_a, dev_c, 5);

	cudaMemcpy(c,dev_c,size,cudaMemcpyDeviceToHost);

	printf("new c:\n");
	for (int i = 0; i < N; i++)
		printf("c[%d].x : %d  \tc[%d].y : %d\tc[%d].z : %c\n", i, c[i].x, i, c[i].y, i, c[i].z);

	//printf("sizeof(char): %d", sizeof(char));

	cudaFree(&dev_a);
	cudaFree(&dev_c);

	return 0;
}
