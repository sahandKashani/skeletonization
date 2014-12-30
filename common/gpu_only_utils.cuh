#ifndef GPU_ONLY_UTILS_CUH
#define GPU_ONLY_UTILS_CUH

#include "lspbmp.hpp"
#include "utils.hpp"

#define gpuErrchk(ans) (gpuAssert((ans), __FILE__, __LINE__))
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        printf("GPU Error: %s (%s:%d)\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

typedef struct {
    int bottom;
    int right;
} Padding;

void gpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding);
void gpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding, dim3* grid_dim, dim3* block_dim);
void pad_binary_bitmap(Bitmap** image, uint8_t binary_padding_value, Padding padding);
void unpad_binary_bitmap(Bitmap** image, Padding padding);

#endif
