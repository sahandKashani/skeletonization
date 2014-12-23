#ifndef GPU_ONLY_UTILS_CUH
#define GPU_ONLY_UTILS_CUH

#include "lspbmp.hpp"
#include "utils.hpp"

#define MAX_THREADS_PER_BLOCK 1024

#define PAD_TOP (2)
#define PAD_LEFT (2)
#define PAD_BOTTOM (1)
#define PAD_RIGHT (1)

typedef struct {
    unsigned int top;
    unsigned int bottom;
    unsigned int left;
    unsigned int right;
} Padding;

void gpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding);
void gpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding, dim3* grid_dim, dim3* block_dim);
void pad_binary_bitmap(Bitmap** image, uint8_t binary_padding_value, Padding padding);
void unpad_binary_bitmap(Bitmap** image, Padding padding);

#endif
