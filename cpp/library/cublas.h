#include <cuda.h>


typedef enum cublasstatus { CUBLAS_STATUS_SUCCESS,
	CUBLAS_STATUS_NOT_INITIALIZED, 
	CUBLAS_STATUS_ALLOC_FAILED,
	CUBLAS_STATUS_INVALID_VALUE,
	CUBLAS_STATUS_ARCH_MISMATCH,
	CUBLAS_STATUS_MAPPING_ERROR,
	CUBLAS_STATUS_EXECUTION_FAILED,
	CUBLAS_STATUS_INTERNAL_ERROR,
	CUBLAS_STATUS_NOT_SUPPORTED,
	CUBLAS_STATUS_LICENSE_ERROR} custatusResult;

typedef enum cublasstatus cublasStatus_t;
typedef struct cublashandle {
} cublasHandle_t;


cublasStatus_t cublasCreate(cublasHandle_t *handle) {
/*
This function initializes the CUBLAS library and creates a handle to an opaque structure holding the CUBLAS library context. It allocates hardware resources on the host and device and must be called prior to making any other CUBLAS library calls. The CUBLAS library context is tied to the current CUDA device. To use the library on multiple devices, one CUBLAS handle needs to be created for each device. Furthermore, for a given device, multiple CUBLAS handles with different configuration can be created. Because cublasCreate allocates some internal resources and the release of those resources by calling cublasDestroy will implicitly call cublasDeviceSynchronize, it is recommended to minimize the number of cublasCreate/cublasDestroy occurences. For multi-threaded applications that use the same device from different threads, the recommended programming model is to create one CUBLAS handle per thread and use that CUBLAS handle for the entire life of the thread. 
*/
	return CUBLAS_STATUS_SUCCESS;
}




cublasStatus_t cublasDestroy(cublasHandle_t handle) {

/*
This function releases hardware resources used by the CUBLAS library. This function is usually the last call with a particular handle to the CUBLAS library. Because cublasCreate allocates some internal resources and the release of those resources by calling cublasDestroy will implicitly call cublasDeviceSynchronize, it is recommended to minimize the number of cublasCreate/cublasDestroy occurences. 
*/

	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize,
                const void *A, int lda, void *B, int ldb) {
/*	This function copies a tile of rows x cols elements from a matrix A in host memory space to a matrix B in GPU memory space. It is assumed that each element requires storage of elemSize bytes and that both matrices are stored in column-major format, with the leading dimension of the source matrix A and destination matrix B given in lda and ldb, respectively. The leading dimension indicates the number of rows of the allocated matrix, even if only a submatrix of it is being used. In general, B is a device pointer that points to an object, or part of an object, that was allocated in GPU memory space via cublasAlloc().
*/ 

	
	
	return CUBLAS_STATUS_SUCCESS;
}


cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize,
                const void *A, int lda, void *B, int ldb) {
/*
This function copies a tile of rows x cols elements from a matrix A in GPU memory space to a matrix B in host memory space. It is assumed that each element requires storage of elemSize bytes and that both matrices are stored in column-major format, with the leading dimension of the source matrix A and destination matrix B given in lda and ldb, respectively. The leading dimension indicates the number of rows of the allocated matrix, even if only a submatrix of it is being used. In general, A is a device pointer that points to an object, or part of an object, that was allocated in GPU memory space via cublasAlloc(). 
*/

	return CUBLAS_STATUS_NOT_INITIALIZED;

}

cublasStatus_t  cublasSscal(cublasHandle_t handle, int n,
                            const float           *alpha,
                            float           *x, int incx) {


	return CUBLAS_STATUS_SUCCESS;
}
