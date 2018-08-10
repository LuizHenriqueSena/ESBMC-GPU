//xfail:BOOGIE_ERROR
//error: possible null pointer access

#include <stdio.h>
#include <cublas.h>

#define N 10//64



int main(){

	float* v;
	float* dev_v;
	float* alpha;
	float aux = 3;

	//initializing cublas handle
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	/* sets the size of v */
	v = (float*)malloc(N*sizeof(float));

	for (int i = 0; i < N ; ++i){
		v[i] = i;
}	

	cudaMalloc((void**)&dev_v, N*sizeof(float)); /* visible only by GPU: __global__ functions */
	cudaMalloc((void**)&alpha, sizeof(float));

	cudaMemcpy(dev_v, v, N*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(alpha, &aux, N*sizeof(float), cudaMemcpyHostToDevice);
	//This function scales the vector dev_v by the factor alpha
	cublasSscal(cublasHandle, N, alpha, dev_v, 1);

		//foor<<<1, N>>>(dev_v, N, c);
		//ESBMC_verify_kernel_fuintt(foor,1,N, dev_v, N, c);
		
	//cudaMemcpy(v, dev_v, dimensionx*dimensiony*sizeof(float), cudaMemcpyDeviceToHost);


	free(v);
	cudaFree(dev_v);

	return 0;
}
