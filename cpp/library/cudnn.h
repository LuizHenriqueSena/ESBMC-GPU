#include <cuda.h>
#include <cuda_powf.h>


typedef struct cudnnactivationdescriptor{
}cudnnActivationDescriptor_t;

typedef struct cudnnhandle {
} cudnnHandle_t;

typedef enum cudnnstatus {
	CUDNN_STATUS_SUCCESS,
	CUDNN_STATUS_NOT_INITIALIZED,
	CUDNN_STATUS_ALLOC_FAILED,
	CUDNN_STATUS_BAD_PARAM,
	CUDNN_STATUS_ARCH_MISMATCH,
	CUDNN_STATUS_MAPPING_ERROR,
	CUDNN_STATUS_EXECUTION_FAILED,
	CUDNN_STATUS_INTERNAL_ERROR,
	CUDNN_STATUS_NOT_SUPPORTED,
	CUDNN_STATUS_LICENSE_ERROR,
	CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING,
	CUDNN_STATUS_RUNTIME_IN_PROGRESS,
	CUDNN_STATUS_RUNTIME_FP_OVERFLOW} cudnnstt;

typedef enum cudnnstatus cudnnStatus_t;

typedef struct cudnntensordesc {
} cudnnTensorDescriptor_t;

typedef struct cudnnfilterdesc {
} cudnnFilterDescriptor_t;

typedef struct cudnnconvoldesc {
} cudnnConvolutionDescriptor_t;

typedef struct cudnnconvolbwd {
} cudnnConvolutionBwdDataAlgo_t;

typedef struct cudnnconvbwdfilter {
} cudnnConvolutionBwdFilterAlgo_t;

typedef struct cudnnconvolvfwdalg {
} cudnnConvolutionFwdAlgo_t;

typedef struct cudnnlrndesc {
} cudnnLRNDescriptor_t;

typedef struct cudnndivnormmd {
} cudnnDivNormMode_t;

typedef enum cudnnactivationmode{
	CUDNN_ACTIVATION_SIGMOID,
	CUDNN_ACTIVATION_RELU,
	CUDNN_ACTIVATION_TANH,
	CUDNN_ACTIVATION_CLIPPED_RELU,
	CUDNN_ACTIVATION_ELU,
	CUDNN_ACTIVATION_IDENTITY} cudnnactmode;

typedef enum cudnnactivationmode cudnnActivationMode_t;

typedef enum cudnnnanpropagation{
	CUDNN_NOT_PROPAGATE_NAN,
	CUDNN_PROPAGATE_NAN} cudnnanprop;

typedef enum cudnnnanpropagation cudnnNanPropagation_t;

typedef enum cudnntensorformat{
	CUDNN_TENSOR_NCHW,
	CUDNN_TENSOR_NHWC,
	CUDNN_TENSOR_NCHW_VECT_C} cudnntnsformat;

typedef enum cudnntensorformat cudnnTensorFormat_t;

typedef enum cudnndatatype {
	CUDNN_DATA_FLOAT,
	CUDNN_DATA_DOUBLE,
	CUDNN_DATA_HALF,
	CUDNN_DATA_INT8,
	CUDNN_DATA_UINT8,
	CUDNN_DATA_INT32,
	CUDNN_DATA_INT8x4,
	CUDNN_DATA_UINT8x4} cudnndttype;

typedef enum cudnndatatype cudnnDataType_t;


float sigmoidFunction(float u) {
	float result;
	__ESBMC_assume(u < 0);
	result = (1/(1 + powf(2.718281,(u*(-1)))));

	return result;
}

float dSigmoidFunction(float u){
	float result;
	result = sigmoidFunction(u)*(1 - sigmoidFunction(u));
	return result;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t tensordesc){
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t actvdesc){
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t * cudnnhan) {
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t     format,
    cudnnDataType_t         dataType,
    int                     n,
    int                     c,
    int                     h,
    int                     w) {
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor(
    cudnnActivationDescriptor_t         activationDesc,
    cudnnActivationMode_t               mode,
    cudnnNanPropagation_t               reluNanOpt,
    double                              coef) {
	return CUDNN_STATUS_SUCCESS;
}


cudnnStatus_t cudnnDestroy(cudnnHandle_t cudnnhan) {
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc){
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc){
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t     activationDesc,
    const float                     *alpha,
    const cudnnTensorDescriptor_t   xDesc,
    const float                     *x,
    const float                     *beta,
    const cudnnTensorDescriptor_t   yDesc,
    float                           *y) {

	int i = 0;
	int limit = 5;
	
	for(i = 0; i<limit; i++) {
	y[i] = sigmoidFunction(x[i]);
}
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationBackward(
    cudnnHandle_t                    handle,
    cudnnActivationDescriptor_t      activationDesc,
    const float                      *alpha,
    const cudnnTensorDescriptor_t    yDesc,
    const float                      *y,
    const cudnnTensorDescriptor_t    dyDesc,
    const float                      *dy,
    const cudnnTensorDescriptor_t    xDesc,
    const float                      *x,
    const float                      *beta,
    const cudnnTensorDescriptor_t    dxDesc,
    float                            *dx) {

	int contador = 0;
	for(contador = 0; contador< 5; contador++){
		dx[contador] = (y[contador]*(1 - y[contador]))*dy[contador];
	}
	return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnAddTensor(
    cudnnHandle_t                     handle,
    const void                       *alpha,
    const cudnnTensorDescriptor_t     aDesc,
    const void                       *A,
    const void                       *beta,
    const cudnnTensorDescriptor_t     cDesc,
    void                             *C) {

}


cudnnStatus_t cudnnConvolutionBackwardBias(
    cudnnHandle_t                    handle,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    dyDesc,
    const void                      *dy,
    const void                      *beta,
    const cudnnTensorDescriptor_t    dbDesc,
    void                            *db) {

}

cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t                       handle,
    const void                         *alpha,
    const cudnnFilterDescriptor_t       wDesc,
    const void                         *w,
    const cudnnTensorDescriptor_t       dyDesc,
    const void                         *dy,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionBwdDataAlgo_t       algo,
    void                               *workSpace,
    size_t                              workSpaceSizeInBytes,
    const void                         *beta,
    const cudnnTensorDescriptor_t       dxDesc,
    void                               *dx) {

}

cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t                       handle,
    const void                         *alpha,
    const cudnnTensorDescriptor_t       xDesc,
    const void                         *x,
    const cudnnTensorDescriptor_t       dyDesc,
    const void                         *dy,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionBwdFilterAlgo_t     algo,
    void                               *workSpace,
    size_t                              workSpaceSizeInBytes,
    const void                         *beta,
    const cudnnFilterDescriptor_t       dwDesc,
    void                               *dw) {

}

cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t                       handle,
    const void                         *alpha,
    const cudnnTensorDescriptor_t       xDesc,
    const void                         *x,
    const cudnnFilterDescriptor_t       wDesc,
    const void                         *w,
    const cudnnConvolutionDescriptor_t  convDesc,
    cudnnConvolutionFwdAlgo_t           algo,
    void                               *workSpace,
    size_t                              workSpaceSizeInBytes,
    const void                         *beta,
    const cudnnTensorDescriptor_t       yDesc,
    void                               *y) {

}

 cudnnStatus_t cudnnDivisiveNormalizationBackward(
    cudnnHandle_t                    handle,
    cudnnLRNDescriptor_t             normDesc,
    cudnnDivNormMode_t               mode,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    xDesc,
    const void                      *x,
    const void                      *means,
    const void                      *dy,
    void                            *temp,
    void                            *temp2,
    const void                      *beta,
    const cudnnTensorDescriptor_t    dxDesc,
    void                            *dx,
    void                            *dMeans) {

}

cudnnStatus_t cudnnDivisiveNormalizationForward(
    cudnnHandle_t                    handle,
    cudnnLRNDescriptor_t             normDesc,
    cudnnDivNormMode_t               mode,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    xDesc,
    const void                      *x,
    const void                      *means,
    void                            *temp,
    void                            *temp2,
    const void                      *beta,
    const cudnnTensorDescriptor_t    yDesc,
    void                            *y) {

}

