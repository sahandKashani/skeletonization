#ifndef GPU1_CUH
#define GPU1_CUH

#include <stdint.h>
#include "../common/utils.hpp"

__device__ uint8_t black_neighbors_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width);
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding padding, dim3 grid_dim, dim3 block_dim);
__global__ void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, unsigned int height, Padding padding);
__device__ uint8_t wb_transitions_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width);

#endif
