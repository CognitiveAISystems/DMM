#pragma once
#include <cuda_runtime.h>

__device__ __forceinline__ bool bit_test_and_set(unsigned* bitmap, int idx) {
    unsigned mask = 1u << (idx & 31);
    unsigned old  = atomicOr(&bitmap[idx >> 5], mask);
    return !(old & mask);
}
