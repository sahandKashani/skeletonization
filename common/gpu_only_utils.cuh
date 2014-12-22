#ifndef GPU_ONLY_UTILS_CUH
#define GPU_ONLY_UTILS_CUH

#include "lspbmp.hpp"
#include "utils.hpp"

#define MAX_THREADS_PER_BLOCK 1024

void gpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding);
void gpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding, dim3* grid_dim, dim3* block_dim);

#endif
