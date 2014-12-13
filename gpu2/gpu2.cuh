#ifndef GPU2_CUH
#define GPU2_CUH

#include <stdint.h>
#include "../common/utils.hpp"

void and_reduction(dim3 grid_dim, dim3 block_dim, uint8_t* d_pixel_equ, uint8_t* d_block_equ, uint8_t* d_grid_equ, unsigned int pixel_equ_size, unsigned int block_equ_size);
__global__ void and_reduction(uint8_t* d_in, uint8_t* d_out, unsigned int size);
__device__ uint8_t black_neighbors_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width);
__global__ void pixel_equality(uint8_t* d_in_1, uint8_t* d_in_2, uint8_t* d_out, unsigned int width, Padding padding);
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding padding, dim3 grid_dim, dim3 block_dim);
__global__ void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, Padding padding);
__device__ uint8_t wb_transitions_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width);

#endif
