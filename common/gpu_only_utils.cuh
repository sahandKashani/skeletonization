#ifndef GPU_ONLY_UTILS_CUH
#define GPU_ONLY_UTILS_CUH

#include <stdint.h>
#include <stdio.h>
#include "lspbmp.hpp"

#define gpuErrchk(ans) (gpuAssert((ans), __FILE__, __LINE__))
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        printf("GPU Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void gpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap);
void gpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3* grid_dim, dim3* block_dim);
uint8_t is_power_of_2(uint8_t x);
int next_power_of_2(int x);

#endif
