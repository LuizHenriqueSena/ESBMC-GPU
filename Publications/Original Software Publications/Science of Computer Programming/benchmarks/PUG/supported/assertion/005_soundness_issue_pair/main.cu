#ifndef _KERNEL_H_ 
#define _KERNEL_H_

#include "config.h"
#include "my_cutil.h"

__global__ void kernel(float * __restrict p1, float * __restrict p2, float * __restrict p3, float * __restrict p4, float * __restrict p5, int x, int y)
{
	funcType fp = a;

    switch(x) {
    case 1:
        fp = &a;
        break;
    case 2:
        fp = &b;
        break;
    case 3:
        fp = &c;
        break;
    case 4:
        fp = &d;
        break;
    default:
        fp = &e;
        break;
    }

    switch(y) {
    case 1:
        fp(p1);
        break;
    case 2:
        fp(p2);
        break;
    case 3:
        fp(p3);
        break;
    case 4:
        fp(p4);
        break;
    default:
        fp(p5);
        break;
    }

   __assert(1);
}

#endif
