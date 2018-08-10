//xfail:BOOGIE_ERROR
//error: possible null pointer access

#include <stdio.h>
#include <cublas.h>

#define N 9//64



int main(){

	double* v;
	double* v2;	
	double* dev_v;
	double* dev_v2;
	double* dev_v3;
	double* alpha;
	double* beta;
	double al = 1;
	double be = 0;
	//double* dev_result;
	
	//initializing cublas handle
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	/* sets the size of v */
	v = (double*)malloc(N*sizeof(double));
	v2 = (double*)malloc(N*sizeof(double));

	for (int i = 0; i < N ; ++i){
		v[i] = i;
}	
	for (int i = 0; i < N; ++i) {
		v2[i] = i*i;
}
		
	cudaMalloc((void**)&dev_v, N*sizeof(double)); /* visible only by GPU: __global__ functions */
	cudaMalloc((void**)&dev_v2, N*sizeof(double));
	cudaMalloc((void**)&dev_v3, N*sizeof(double));
	cudaMalloc((void**)&alpha, sizeof(double));
	//cudaMalloc((void**)&beta, sizeof(double));

	cudaMemcpy(dev_v, v, N*sizeof(double), cudaMemcpyHostToDevice);	
	cudaMemcpy(dev_v2, v2, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(alpha, &al, sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(beta, &be, sizeof(double), cudaMemcpyHostToDevice);
	//This function operates the dev_v3= alpha*(dev_v) x dev_v2 + beta*dev_v3

	cublasDaxpy(cublasHandle, N,
                           alpha,
                           dev_v, 1,
                           dev_v2, 1);
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
