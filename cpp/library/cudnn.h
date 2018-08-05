#include <cuda.h>

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
	CUDNN_STATUS_RUNTIME_IN_PROGRESS,s
	CUDNN_STATUS_RUNTIME_FP_OVERFLOW} cudnnstt;

typedef enum cudnnstatus cudnnStatus_t;

typedef struct cudnntensordesc {
} cudnnTensorDescriptor_t;

typedef struct cudnnactvdesc {
} cudnnActivationDescriptor_t;

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

void cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t tensordesc){
	return CUDNN_STATUS_SUCCESS;
}

void cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t actvdesc){
	return CUDNN_STATUS_SUCCESS;
}

void cudnnStatus_t cudnnCreate(cudnnHandle_t cudnnhan) {
	return CUDNN_STATUS_SUCCESS;
}

void cudnnStatus_t cudnnSetTensor4dDescriptor(
    cudnnTensorDescriptor_t tensorDesc,
    cudnnTensorFormat_t     format,
    cudnnDataType_t         dataType,
    int                     n,
    int                     c,
    int                     h,
    int                     w) {
	return CUDNN_STATUS_SUCCESS;
}

void cudnnStatus_t cudnnSetActivationDescriptor(
    cudnnActivationDescriptor_t         activationDesc,
    cudnnActivationMode_t               mode,
    cudnnNanPropagation_t               reluNanOpt,
    double                              coef) {
	return CUDNN_STATUS_SUCCESS;
}


void cudnnStatus_t cudnnDestroy(cudnnHandle_t cudnnhan) {
	return CUDNN_STATUS_SUCCESS;
}

void cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc){
	return CUDNN_STATUS_SUCCESS;
}

void cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc){
	return CUDNN_STATUS_SUCCESS;
}

void cudnnStatus_t cudnnActivationForward(
    cudnnHandle_t handle,
    cudnnActivationDescriptor_t     activationDesc,
    const void                     *alpha,
    const cudnnTensorDescriptor_t   xDesc,
    const void                     *x,
    const void                     *beta,
    const cudnnTensorDescriptor_t   yDesc,
    void                           *y) {

	int i = 0;
	



































