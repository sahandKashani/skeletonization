#ifndef GPU2_CUH
#define GPU2_CUH

#include <stdint.h>
#include "../common/utils.hpp"

void and_reduction(dim3 grid_dim, dim3 block_dim, uint8_t* d_pixel_equ, int pixel_equ_size);
__global__ void and_reduction(uint8_t* d_data, int size);
__device__ uint8_t black_neighbors_around(uint8_t* d_data, int row, int col, int width, int height);
__device__ uint8_t is_outside_image(int row, int col, int width, int height);
__device__ uint8_t P2_f(uint8_t* data, int row, int col, int width, int height);
__device__ uint8_t P3_f(uint8_t* data, int row, int col, int width, int height);
__device__ uint8_t P4_f(uint8_t* data, int row, int col, int width, int height);
__device__ uint8_t P5_f(uint8_t* data, int row, int col, int width, int height);
__device__ uint8_t P6_f(uint8_t* data, int row, int col, int width, int height);
__device__ uint8_t P7_f(uint8_t* data, int row, int col, int width, int height);
__device__ uint8_t P8_f(uint8_t* data, int row, int col, int width, int height);
__device__ uint8_t P9_f(uint8_t* data, int row, int col, int width, int height);
__global__ void pixel_equality(uint8_t* d_in_1, uint8_t* d_in_2, uint8_t* d_out, int width, int height);
int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3 grid_dim, dim3 block_dim);
__global__ void skeletonize_pass(uint8_t* d_src, uint8_t* d_dst, int width, int height);
__device__ uint8_t wb_transitions_around(uint8_t* d_data, int row, int col, int width, int height);

#endif
