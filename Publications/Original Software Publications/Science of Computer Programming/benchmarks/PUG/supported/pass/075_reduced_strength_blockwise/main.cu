#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(int *A) {
//  __assert(blockDim.x <= WIDTH);
//#ifdef BLOCK_DIVIDES_WIDTH
//  //__assert(__mod_pow2(WIDTH, blockDim.x) == 0);
//#endif

  for (int i=threadIdx.x; i<WIDTH; i+=blockDim.x) {

//#ifndef BLOCK_DIVIDES_WIDTH
//    // working set(1) using global invariants
//    /*A*/__global_invariant(__write_implies(A, (blockIdx.x*WIDTH) <= __write_offset_bytes(A)/sizeof(int))),
//    /*B*/__global_invariant(__write_implies(A,                       __write_offset_bytes(A)/sizeof(int) < (blockIdx.x+1)*WIDTH)),
//    /*C*/__invariant(threadIdx.x <= i),
//    /*D*/__invariant(               i <= WIDTH+blockDim.x),
//         __invariant(i % blockDim.x == threadIdx.x),
//         __global_invariant(__write_implies(A, (((__write_offset_bytes(A)/sizeof(int)) % WIDTH) % blockDim.x) == threadIdx.x)),
//#else
//    // working set(2) iff WIDTH % blockDim.x == 0
//    /*A*/__invariant(__write_implies(A, (blockIdx.x*WIDTH) <= __write_offset_bytes(A)/sizeof(int))),
//    /*B*/__invariant(__write_implies(A,                       __write_offset_bytes(A)/sizeof(int) < (blockIdx.x+1)*WIDTH)),
//    /*C*/__invariant(threadIdx.x <= i),
//    /*D*/__invariant(               i <= WIDTH+blockDim.x),
//         __invariant(__uniform_int((i-threadIdx.x))),
//         __invariant(__uniform_bool(__enabled())),
//#endif

    A[blockIdx.x*WIDTH+i] = i;
  }

//#ifdef FORCE_FAIL
//  __assert(false);
//#endif
}

#endif
