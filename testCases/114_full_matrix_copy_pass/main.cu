//xfail:BOOGIE_ERROR
//error: possible null pointer access

#include <stdio.h>
#include <cublas.h>

#define N 10//64



int main(){

	float* v;
	float* dev_v;
	int dimensionx = 5;
	int dimensiony = 2;

	//initializing cublas handle
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	/* sets the size of v */
	v = (float*)malloc(dimensionx*dimensiony*sizeof(float));

	for (int i = 0; i <dimensionx*dimensiony ; ++i){
		v[i] = i;
}	

	cudaMalloc((void**)&dev_v, dimensionx*dimensiony*sizeof(float)); /* visible only by GPU: __global__ functions */

	cublasSetMatrix(dimensionx, dimensiony, sizeof(float), dev_v, dimensiony, v, dimensionx);	

		//foor<<<1, N>>>(dev_v, N, c);
		//ESBMC_verify_kernel_fuintt(foor,1,N, dev_v, N, c);
		
	//cudaMemcpy(v, dev_v, dimensionx*dimensiony*sizeof(float), cudaMemcpyDeviceToHost);

	//cublasGetMatrix(dimensionx, dimensiony, sizeof(float), v, dimensiony, dev_v, dimensiony);

	free(v);
	cudaFree(dev_v);

	return 0;
}
