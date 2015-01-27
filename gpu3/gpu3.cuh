#ifndef GPU3_CUH
#define GPU3_CUH

#include <stdint.h>
#include "../common/lspbmp.hpp"

void and_reduction(uint8_t* g_src_data, uint8_t* g_dst_data, uint8_t* g_equ_data, int g_width, int g_height, dim3 grid_dim, dim3 block_dim);
__global__ void and_reduction(uint8_t* g_data, int g_size);
__device__ uint8_t black_neighbors_around(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t block_and_reduce(uint8_t* s_data);
__device__ uint8_t border_global_mem_read(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height);
__device__ void border_global_mem_write(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height, uint8_t write_data);
__device__ uint8_t is_outside_image(int g_row, int g_col, int g_width, int g_height);
__device__ void load_s_src(uint8_t* g_src, int g_row, int g_col, int g_width, int g_height, uint8_t* s_src, int s_row, int s_col, int s_width);
__device__ uint8_t P2_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P3_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P4_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P5_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P6_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P7_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P8_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__device__ uint8_t P9_f(uint8_t* s_data, int s_row, int s_col, int s_width);
__global__ void pixel_equality(uint8_t* g_in_1, uint8_t* g_in_2, uint8_t* g_out, int g_width, int g_height);
int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3 grid_dim, dim3 block_dim);
__global__ void skeletonize_pass(uint8_t* g_src, uint8_t* g_dst, int g_width, int g_height);
__device__ uint8_t wb_transitions_around(uint8_t* s_data, int s_row, int s_col, int s_width);

#endif
