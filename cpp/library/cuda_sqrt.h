#include <stdint.h>
#include <limits.h>
#include <cuda_builtin_clz.h>

typedef double db_t;
typedef uint64_t dbrep_t;
static const int significandBitssqrt = 52;
#define REP_C_sqrt UINT64_C

static inline int rep_clz_sqrt(dbrep_t a) {
    if (a & REP_C_sqrt(0xffffffff00000000))
        return 32 + __builtin_clz(a >> 32);
    else 
        return __builtin_clz(a & REP_C_sqrt(0xffffffff));
}

static inline dbrep_t toRep_sqrt(db_t x) {
    union { dbrep_t i; db_t f; } rep;
	rep.f = x;
    return rep.i;
}

static inline db_t fromRep_Sqrt(dbrep_t x) {
    union { dbrep_t i; db_t f; } rep;
	rep.i = x;
    return rep.f;
}

static inline uint32_t mulhi_sqrt(uint32_t a, uint32_t b) {
    return (uint64_t)a*b >> 32;
}

#define loWord(a) (a & 0xffffffffU)
#define hiWord(a) (a >> 32)
static inline uint64_t mulhi64(uint64_t a, uint64_t b) {
    // Each of the component 32x32 -> 64 products
    const uint64_t plolo = loWord(a) * loWord(b);
    const uint64_t plohi = loWord(a) * hiWord(b);
    const uint64_t philo = hiWord(a) * loWord(b);
    const uint64_t phihi = hiWord(a) * hiWord(b);
    // Sum terms that contribute to lo in a way that allows us to get the carry
    const uint64_t r0 = loWord(plolo);
    const uint64_t r1 = hiWord(plolo) + loWord(plohi) + loWord(philo);
    // Sum terms contributing to hi with the carry from lo
    return phihi + hiWord(plohi) + hiWord(philo) + hiWord(r1);
}

db_t sqrt(db_t x) {
    
    // Various constants parametrized by the type of x:
     int typeWidth = sizeof(dbrep_t) * CHAR_BIT;
     int exponentBits = typeWidth - significandBitssqrt - 1;
     int exponentBias = (1 << (exponentBits - 1)) - 1;
     dbrep_t minNormal = REP_C_sqrt(1) << significandBitssqrt;
     dbrep_t significandMask = minNormal - 1;
     dbrep_t signBit = REP_C_sqrt(1) << (typeWidth - 1);
     dbrep_t absMask = signBit - 1;
     dbrep_t infRep = absMask ^ significandMask;
     dbrep_t qnan = infRep | REP_C_sqrt(1) << (significandBitssqrt - 1);
    
    // Extract the various important bits of x
    const dbrep_t xRep = toRep_sqrt(x);
    dbrep_t significand = xRep & significandMask;
    int exponent = (xRep >> significandBitssqrt) - exponentBias;
    
    // Using an unsigned integer compare, we can detect all of the special
    // cases with a single branch: zero, denormal, negative, infinity, or NaN.
    if (xRep - minNormal >= infRep - minNormal) {
        const dbrep_t xAbs = xRep & absMask;
        // sqrt(+/- 0) = +/- 0
        if (xAbs == 0) return x;
        // sqrt(NaN) = qNaN
        if (xAbs > infRep) return fromRep_Sqrt(qnan | xRep);
        // sqrt(negative) = qNaN
        if (xRep > signBit) return fromRep_Sqrt(qnan);
        // sqrt(infinity) = infinity
        if (xRep == infRep) return x;
        
        // normalize denormals and fall back into the mainline
        const int shift = rep_clz_sqrt(significand) - rep_clz_sqrt(minNormal);
        significand <<= shift;
        exponent += 1 - shift;
    }
    
    // Insert the implicit bit of the significand.  If x was denormal, then
    // this bit was already set by the normalization process, but it won't hurt
    // to set it twice.
    significand |= minNormal;
    
    // Halve the exponent to get the exponent of the result, and transform the
    // significand into a Q30 fixed-point xQ30 in the range [1,4) -- if the
    // exponent of x is odd, then xQ30 is in [2,4); if it is even, then xQ30
    // is in [1,2).
    const int resultExponent = exponent >> 1;
    const uint64_t xQ62 = significand << (10 + (exponent & 1));
    const uint32_t xQ30 = xQ62 >> 32;
    
    // Q32 linear approximation to the reciprocal square root of xQ30.  This
    // approximation is good to a bit more than 3.5 bits:
    //
    //     1/sqrt(a) ~ 1.1033542890963095 - a/6
    const uint32_t oneSixthQ34 = UINT32_C(0xaaaaaaaa);
    uint32_t recipQ32 = UINT32_C(0x1a756d3b) - mulhi_sqrt(oneSixthQ34, xQ30);
    
    // Newton-Raphson iterations to improve our reciprocal:
    const uint32_t threeQ30 = UINT32_C(0xc0000000);
    uint32_t residualQ30 = mulhi_sqrt(xQ30, mulhi_sqrt(recipQ32, recipQ32));
    recipQ32 = mulhi_sqrt(recipQ32, threeQ30 - residualQ30) << 1;
    residualQ30 = mulhi_sqrt(xQ30, mulhi_sqrt(recipQ32, recipQ32));
    recipQ32 = mulhi_sqrt(recipQ32, threeQ30 - residualQ30) << 1;
    residualQ30 = mulhi_sqrt(xQ30, mulhi_sqrt(recipQ32, recipQ32));
    recipQ32 = mulhi_sqrt(recipQ32, threeQ30 - residualQ30) << 1;
    residualQ30 = mulhi_sqrt(xQ30, mulhi_sqrt(recipQ32, recipQ32));
    recipQ32 = mulhi_sqrt(recipQ32, threeQ30 - residualQ30) << 1;
    
    // We need to compute the final Newton-Raphson step with 64-bit words:
    const uint64_t threeQ62 = UINT64_C(0xc000000000000000);
    const uint64_t residualQ62 = mulhi64(xQ62,(uint64_t)recipQ32*recipQ32);
    const uint64_t stepQ62 = threeQ62 - residualQ62;
    const uint64_t recipQ63hi = recipQ32 * hiWord(stepQ62);
    const uint64_t recipQ63lo = recipQ32 * loWord(stepQ62) >> 32;
    const uint64_t recipQ64 = (recipQ63hi + recipQ63lo) << 1;
    
    // recipQ64 now holds an approximate 1/sqrt(x).  Multiply by x to get an
    // initial sqrt(x) in Q52.  From the construction of this estimate, we know
    // that it is either the correctly rounded significand of the result or one
    // less than the correctly rounded significand (the -2 guarantees that we
    // fall on the correct side of the actual square root).
    dbrep_t result = (mulhi64(recipQ64, xQ62) - 2) >> 10;
    
    // Compute the residual x - result*result to decide if the result needs to
    // be rounded up.
    dbrep_t residual = (xQ62 << 42) - result*result;
    result += residual > result;
    
    // Clear the implicit bit of result:
    result &= significandMask;
    // Insert the exponent:
    result |= (dbrep_t)(resultExponent + exponentBias) << significandBitssqrt;
    return fromRep_Sqrt(result);    
}
