//xfail:BOOGIE_ERROR
//error: possible null pointer access

#include <stdio.h>
#include <cublas.h>
#include <cudnn.h>

#define N 9//64



int main(){

	float* v;
	float* v2;	
	float* dev_v;
	float* dev_v2;
	float* dev_v3;
	float* dev_dlossdata;
	float* dev_result;
	float* dev_dbackpropagation;
	float* alpha;
	float* beta;
	float al = 1;
	float be = 0;
	//float* dev_result;
	
	//initializing cublas handle
	cublasHandle_t cublasHandle;
	cudnnHandle_t cudnnHandle;
	cublasCreate(&cublasHandle);
	cudnnCreate(&cudnnHandle);

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
	cudaMalloc((void**)&dev_v3, N*sizeof(float));
	cudaMalloc((void**)&dev_dlossdata, N*sizeof(float));
	cudaMalloc((void**)&dev_result, N*sizeof(float));
	cudaMalloc((void**)&dev_dbackpropagation, N*sizeof(float));
	cudaMalloc((void**)&alpha, sizeof(float));
	cudaMalloc((void**)&beta, sizeof(float));

	cudaMemcpy(dev_v, v, N*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy(dev_v2, v2, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(alpha, &al, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(beta, &be, sizeof(float), cudaMemcpyHostToDevice);
	//This function operates the dev_v3= alpha*(dev_v) x dev_v2 + beta*dev_v3


	cudnnActivationDescriptor_t activationDesc;
	cudnnTensorDescriptor_t xDesc, yDesc, dxDesc, dyDesc;
	
	cublasSgemm(cublasHandle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			3, 1, 3,
			alpha,
			dev_v, 3,
			dev_v2, 3,
			beta,
			dev_v3, 3);

	cudnnActivationForward(cudnnHandle, activationDesc, alpha, xDesc, dev_v3, beta, yDesc, dev_result);

	cudnnActivationBackward(cudnnHandle, activationDesc, alpha, yDesc, dev_result, dyDesc, dev_dlossdata, xDesc, dev_v, beta, dxDesc, dev_dbackpropagation);
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
