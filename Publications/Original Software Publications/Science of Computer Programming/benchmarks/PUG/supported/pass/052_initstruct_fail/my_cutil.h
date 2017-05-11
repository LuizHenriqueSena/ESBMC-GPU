#ifndef _MY_CUTIL_H_
#define _MY_CUTIL_H_

// variable modifiers
#define __shared__ volatile
#define __device__ volatile
#define __constant__ const

// function modifiers
#define __global__ static
#define __kernel__ inline
#define __host__ extern

typedef struct { char x; } char1;
typedef struct { unsigned char x; } uchar1;
typedef struct { short x; } short1;
typedef struct { unsigned short x; } ushort1;
typedef struct { int x; } int1;
typedef struct { unsigned int x; } uint1;
typedef struct { long x; } long1;
typedef struct { unsigned long x; } ulong1;
typedef struct { float x; } float1;
typedef struct { char x, y; } char2;
typedef struct { unsigned char x, y; } uchar2;
typedef struct { short x, y; } short2;
typedef struct { unsigned short x, y; } ushort2;
typedef struct { int x, y; } int2;
typedef struct { unsigned int x, y; } uint2;
typedef struct { long x, y; } long2;
typedef struct { unsigned long x, y; } ulong2;
typedef struct { float x, y; } float2;
typedef struct { char x, y, z; } char3;
typedef struct { unsigned char x, y, z; } uchar3;
typedef struct { short x, y, z; } short3;
typedef struct { unsigned short x, y, z; } ushort3;
typedef struct { int x, y, z; } int3;
typedef struct { unsigned int x, y, z; } uint3;
typedef struct { long x, y, z; } long3;
typedef struct { unsigned long x, y, z; } ulong3;
typedef struct { float x, y, z; } float3;
typedef struct { char x, y, z, w; } char4;
typedef struct { unsigned char x, y, z, w; } uchar4;
typedef struct { short x, y, z, w; } short4;
typedef struct { unsigned short x, y, z, w; } ushort4;
typedef struct { int x, y, z, w; } int4;
typedef struct { unsigned int x, y, z, w; } uint4;
typedef struct { long x, y, z, w; } long4;
typedef struct { unsigned long x, y, z, w; } ulong4;
typedef struct { float x, y, z, w; } float4;

typedef uint3 dim3;
typedef char CUTBoolean;
extern dim3 gridDim;
extern uint3 blockIdx;
extern dim3 blockDim;
extern uint3 threadIdx;
extern int cudaMemcpyHostToDevice;
extern int cudaMemcpyDeviceToHost;
extern void __syncthreads();

extern uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w);

extern void assert(int expression);
extern void assume(int expression);
extern int is_sv(int exp);

extern unsigned int threadIdx_x;
extern unsigned int threadIdx_y;
extern unsigned int threadIdx_z;

#define atomicAdd(x,y) x * y
#define bar __syncthreads
#define _mul24(x,y) x * y
#define __umul24(x,y) x * y

#define RACE(x) x[1] = 1             // enforce a race checking

#endif
