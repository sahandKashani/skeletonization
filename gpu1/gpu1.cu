#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu1.cuh"
#include "../common/gpu_only_utils.cuh"
#include "../common/lspbmp.hpp"
#include "../common/utils.hpp"

// Computes the number of black neighbors around a pixel.
__device__ uint8_t black_neighbors_around(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    uint8_t count = 0;

    count += (P2_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P3_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P4_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P5_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P6_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P7_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P8_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);
    count += (P9_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK);

    return count;
}

__device__ uint8_t global_mem_read(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return is_outside_image(g_row, g_col, g_width, g_height) ? BINARY_WHITE : g_data[g_row * g_width + g_col];
}

__device__ void global_mem_write(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height, uint8_t write_data) {
    if (!is_outside_image(g_row, g_col, g_width, g_height)) {
        g_data[g_row * g_width + g_col] = write_data;
    }
}

__device__ uint8_t is_outside_image(int g_row, int g_col, int g_width, int g_height) {
    return (g_row < 0) | (g_row > (g_height - 1)) | (g_col < 0) | (g_col > (g_width - 1));
}

__device__ uint8_t P2_f(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return global_mem_read(g_data, g_row - 1, g_col, g_width, g_height);
}

__device__ uint8_t P3_f(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return global_mem_read(g_data, g_row - 1, g_col - 1, g_width, g_height);
}

__device__ uint8_t P4_f(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return global_mem_read(g_data, g_row, g_col - 1, g_width, g_height);
}

__device__ uint8_t P5_f(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return global_mem_read(g_data, g_row + 1, g_col - 1, g_width, g_height);
}

__device__ uint8_t P6_f(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return global_mem_read(g_data, g_row + 1, g_col, g_width, g_height);
}

__device__ uint8_t P7_f(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return global_mem_read(g_data, g_row + 1, g_col + 1, g_width, g_height);
}

__device__ uint8_t P8_f(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return global_mem_read(g_data, g_row, g_col + 1, g_width, g_height);
}

__device__ uint8_t P9_f(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    return global_mem_read(g_data, g_row - 1, g_col + 1, g_width, g_height);
}

// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3 grid_dim, dim3 block_dim) {
    // allocate memory on device
    uint8_t* g_src_data = NULL;
    uint8_t* g_dst_data = NULL;
    int g_data_size = (*src_bitmap)->width * (*src_bitmap)->height * sizeof(uint8_t);
    gpuErrchk(cudaMalloc((void**) &g_src_data, g_data_size));
    gpuErrchk(cudaMalloc((void**) &g_dst_data, g_data_size));

    // send data to device
    gpuErrchk(cudaMemcpy(g_src_data, (*src_bitmap)->data, g_data_size, cudaMemcpyHostToDevice));

    int iterations = 0;
    do {
        skeletonize_pass<<<grid_dim, block_dim>>>(g_src_data, g_dst_data, (*src_bitmap)->width, (*src_bitmap)->height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // bring data back from device
        gpuErrchk(cudaMemcpy((*src_bitmap)->data, g_src_data, g_data_size, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy((*dst_bitmap)->data, g_dst_data, g_data_size, cudaMemcpyDeviceToHost));

        swap_bitmaps((void**) &g_src_data, (void**) &g_dst_data);

        iterations++;
        printf(".");
        fflush(stdout);
    } while (!are_identical_bitmaps(*src_bitmap, *dst_bitmap));

    // free memory on device
    gpuErrchk(cudaFree(g_src_data));
    gpuErrchk(cudaFree(g_dst_data));

    return iterations;
}

// Performs 1 iteration of the thinning algorithm.
__global__ void skeletonize_pass(uint8_t* g_src, uint8_t* g_dst, int g_width, int g_height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int g_size = g_width * g_height;

    while (tid < g_size) {
        int g_row = (tid / g_width);
        int g_col = (tid % g_width);

        uint8_t NZ = black_neighbors_around(g_src, g_row, g_col, g_width, g_height);
        uint8_t TR_P1 = wb_transitions_around(g_src, g_row, g_col, g_width, g_height);
        uint8_t TR_P2 = wb_transitions_around(g_src, g_row - 1, g_col, g_width, g_height);
        uint8_t TR_P4 = wb_transitions_around(g_src, g_row, g_col - 1, g_width, g_height);
        uint8_t P2 = P2_f(g_src, g_row, g_col, g_width, g_height);
        uint8_t P4 = P4_f(g_src, g_row, g_col, g_width, g_height);
        uint8_t P6 = P6_f(g_src, g_row, g_col, g_width, g_height);
        uint8_t P8 = P8_f(g_src, g_row, g_col, g_width, g_height);

        uint8_t thinning_cond_1 = ((2 <= NZ) & (NZ <= 6));
        uint8_t thinning_cond_2 = (TR_P1 == 1);
        uint8_t thinning_cond_3 = (((P2 & P4 & P8) == 0) | (TR_P2 != 1));
        uint8_t thinning_cond_4 = (((P2 & P4 & P6) == 0) | (TR_P4 != 1));
        uint8_t thinning_cond_ok = thinning_cond_1 & thinning_cond_2 & thinning_cond_3 & thinning_cond_4;

        uint8_t g_dst_next = (thinning_cond_ok * BINARY_WHITE) + ((1 - thinning_cond_ok) * global_mem_read(g_src, g_row, g_col, g_width, g_height));
        global_mem_write(g_dst, g_row, g_col, g_width, g_height, g_dst_next);

        tid += (gridDim.x * blockDim.x);
    }
}

// Computes the number of white to black transitions around a pixel.
__device__ uint8_t wb_transitions_around(uint8_t* g_data, int g_row, int g_col, int g_width, int g_height) {
    uint8_t count = 0;

    count += ((P2_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P3_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P3_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P4_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P4_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P5_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P5_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P6_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P6_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P7_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P7_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P8_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P8_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P9_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));
    count += ((P9_f(g_data, g_row, g_col, g_width, g_height) == BINARY_WHITE) & (P2_f(g_data, g_row, g_col, g_width, g_height) == BINARY_BLACK));

    return count;
}

int main(int argc, char** argv) {
    Bitmap* src_bitmap = NULL;
    Bitmap* dst_bitmap = NULL;
    Padding padding_for_thread_count;
    dim3 grid_dim;
    dim3 block_dim;

    gpu_pre_skeletonization(argc, argv, &src_bitmap, &dst_bitmap, &padding_for_thread_count, &grid_dim, &block_dim);

    int iterations = skeletonize(&src_bitmap, &dst_bitmap, grid_dim, block_dim);
    printf(" %u iterations\n", iterations);
    printf("\n");

    gpu_post_skeletonization(argv, &src_bitmap, &dst_bitmap, padding_for_thread_count);

    return EXIT_SUCCESS;
}
