//xfail:BOOGIE_ERROR
//error: possible null pointer access

#include <stdio.h>
#include <cublas.h>

#define N 10//64



int main(){

	float* v;
	float* v2;	
	float* dev_v;
	float* dev_v2;
	//float* dev_result;
	
	//initializing cublas handle
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	/* sets the size of v */
	v = (float*)malloc(N*sizeof(float));
	v2 = (float*)malloc(N*sizeof(float));

	for (int i = 0; i < N ; ++i){
		v[i] = i;
}	
	for (int i = 0; i < N; ++i) {
		v2[i] = i*i;
}
		
	cudaMalloc((void**)&dev_v, N*sizeof(float)); /* visible only by GPU: __global__ functions */
	cudaMalloc((void**)&dev_v2, N*sizeof(float));
	//cudaMalloc((void**)&dev_result, sizeof(float));

	cudaMemcpy(dev_v, v, N*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(dev_v2, v2, N*sizeof(float), cudaMemcpyHostToDevice);
	//This function copies the vector dev_v to dev_v2
	cublasScopy(cublasHandle, N, dev_v, 1, dev_v2, 1);
		//foor<<<1, N>>>(dev_v, N, c);
		//ESBMC_verify_kernel_fuintt(foor,1,N, dev_v, N, c);
		
	//cudaMemcpy(v, dev_v, dimensionx*dimensiony*sizeof(float), cudaMemcpyDeviceToHost);


	free(v);
	free(v2);
	cudaFree(dev_v);
	cudaFree(dev_v2);
	//cudaFree(dev_result);

	return 0;
}
