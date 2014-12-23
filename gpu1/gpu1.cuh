#ifndef GPU1_CUH
#define GPU1_CUH

#include <stdint.h>
#include "../common/gpu_only_utils.cuh"

__device__ uint8_t black_neighbors_around(uint8_t* d_data, int row, int col, unsigned int width, unsigned int height, unsigned int iterations);
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3 grid_dim, dim3 block_dim);
__global__ void skeletonize_pass(uint8_t* d_src, uint8_t* d_dst, unsigned int width, unsigned int height, unsigned int iterations);
__device__ uint8_t wb_transitions_around(uint8_t* d_data, int row, int col, unsigned int width, unsigned int height, unsigned int iterations);

#endif
