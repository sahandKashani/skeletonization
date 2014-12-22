#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu4.cuh"
#include "../common/gpu_only_utils.cuh"
#include "../common/utils.hpp"

#define P2(d_data, row, col, width) ((d_data)[((row) - 1) * (width) +  (col)     ])
#define P3(d_data, row, col, width) ((d_data)[((row) - 1) * (width) + ((col) - 1)])
#define P4(d_data, row, col, width) ((d_data)[ (row)      * (width) + ((col) - 1)])
#define P5(d_data, row, col, width) ((d_data)[((row) + 1) * (width) + ((col) - 1)])
#define P6(d_data, row, col, width) ((d_data)[((row) + 1) * (width) +  (col)     ])
#define P7(d_data, row, col, width) ((d_data)[((row) + 1) * (width) + ((col) + 1)])
#define P8(d_data, row, col, width) ((d_data)[ (row)      * (width) + ((col) + 1)])
#define P9(d_data, row, col, width) ((d_data)[((row) - 1) * (width) + ((col) + 1)])

// Computes the number of black neighbors around a pixel.
__device__ uint8_t black_neighbors_around(uint8_t* d_data, unsigned int row, unsigned int col, unsigned int width) {
    uint8_t count = 0;

    count += (P2(d_data, row, col, width) == BINARY_BLACK);
    count += (P3(d_data, row, col, width) == BINARY_BLACK);
    count += (P4(d_data, row, col, width) == BINARY_BLACK);
    count += (P5(d_data, row, col, width) == BINARY_BLACK);
    count += (P6(d_data, row, col, width) == BINARY_BLACK);
    count += (P7(d_data, row, col, width) == BINARY_BLACK);
    count += (P8(d_data, row, col, width) == BINARY_BLACK);
    count += (P9(d_data, row, col, width) == BINARY_BLACK);

    return count;
}

// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding padding, dim3 grid_dim, dim3 block_dim) {
    // allocate memory on device
    uint8_t* d_src_data = NULL;
    uint8_t* d_dst_data = NULL;
    unsigned int data_size = (*src_bitmap)->width * (*src_bitmap)->height * sizeof(uint8_t);
    cudaError d_src_malloc_success = cudaMalloc((void**) &d_src_data, data_size);
    cudaError d_dst_malloc_success = cudaMalloc((void**) &d_dst_data, data_size);
    assert((d_src_malloc_success == cudaSuccess) && "Error: could not allocate memory for d_src_data");
    assert((d_dst_malloc_success == cudaSuccess) && "Error: could not allocate memory for d_dst_data");

    // send data to device
    cudaMemcpy(d_src_data, (*src_bitmap)->data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_data, (*dst_bitmap)->data, data_size, cudaMemcpyHostToDevice);

    unsigned int iterations = 0;
    do {
        skeletonize_pass<<<grid_dim, block_dim, (block_dim.x + padding.left + padding.right) * (block_dim.y + padding.top + padding.bottom) * sizeof(uint8_t)>>>(d_src_data, d_dst_data, (*src_bitmap)->width, padding);

        // bring data back from device
        cudaMemcpy((*src_bitmap)->data, d_src_data, data_size, cudaMemcpyDeviceToHost);
        cudaMemcpy((*dst_bitmap)->data, d_dst_data, data_size, cudaMemcpyDeviceToHost);

        swap_bitmaps((void**) &d_src_data, (void**) &d_dst_data);

        iterations++;
        printf(".");
        fflush(stdout);
    } while (!are_identical_bitmaps(*src_bitmap, *dst_bitmap));

    // free memory on device
    cudaFree(d_src_data);
    cudaFree(d_dst_data);

    return iterations;
}

// Performs 1 iteration of the thinning algorithm.
__global__ void skeletonize_pass(uint8_t* d_src, uint8_t* d_dst, unsigned int width, Padding padding) {
    // shared memory for d_src tile
    extern __shared__ uint8_t s_src[];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int bx = blockIdx.x;
    unsigned int by = blockIdx.y;
    unsigned int bdx = blockDim.x;
    unsigned int bdy = blockDim.y;

    unsigned int row = by * bdy + ty + padding.top;
    unsigned int col = bx * bdx + tx + padding.left;

    // load a tile of d_src into s_src

    // center-center: Each thread will load 1 value
    s_src[(ty + padding.top) * bdx + (tx + padding.left)] = d_src[row * width + col];

    // corner cases
    if (((tx % bdx) == 0) & ((ty % bdy) == 0)) {
        // top-left: padding.top * padding.left values to load by 1 thread
        for (unsigned int r = 0; r < padding.top; r++) {
            for (unsigned int c = 0; c < padding.left; c++) {
                s_src[r * bdx + c] = d_src[(row - padding.top + r) * width + (col - padding.left + c)];
            }
        }
    } else if (((tx % bdx) == (bdx - 1)) & ((ty % bdy) == 0)) {
        // top-right: padding.top * padding.right values to load by 1 thread
        for (unsigned int r = 0; r < padding.top; r++) {
            for (unsigned int c = 0; c < padding.right; c++) {
                s_src[r * bdx + bdx + c] = d_src[(row - padding.top + r) * width + (col + c + 1)];
            }
        }
    } else if (((tx % bdx) == 0) & ((ty % bdy) == (bdy - 1))) {
        // bottom-left: padding.bottom * padding.left values to load by 1 thread
        for (unsigned int r = 0; r < padding.bottom; r++) {
            for (unsigned int c = 0; c < padding.left; c++) {
                s_src[(padding.top + bdy + r) * bdx + c] = d_src[(row + r + 1) * width + (col - padding.left + c)];
            }
        }
    } else if (((tx % bdx) == (bdx - 1)) & ((ty % bdy) == (bdy - 1))) {
        // bottom-right: padding.bottom * padding.right values to load by 1 thread
        for (unsigned int r = 0; r < padding.bottom; r++) {
            for (unsigned int c = 0; c < padding.right; c++) {
                s_src[(padding.top + bdy + r) * bdx + bdx + c] = d_src[(row + r + 1) * width + (col + c + 1)];
            }
        }
    } else if ((tx % bdx) == 0) {
        // left-center: padding.left values to load PER thread
        for (unsigned int c = 0; c < padding.left; c++) {
            s_src[(ty + padding.top) * bdx + c] = d_src[row * width + c];
        }
    } else if ((tx % bdx) == (bdx - 1)) {
        // right-center: padding.right values to load PER thread
    } else if ((ty % bdy) == 0) {
        // top-center: padding.top values to load PER thread
    } else if ((ty % bdy) == (bdy - 1)) {
        // bottom-center: padding.bottom values to load PER thread
    }

    // make sure all threads have finished loading their data into shared memory
    __syncthreads();

    uint8_t NZ = black_neighbors_around(d_src, row, col, width);
    uint8_t TR_P1 = wb_transitions_around(d_src, row, col, width);
    uint8_t TR_P2 = wb_transitions_around(d_src, row - 1, col, width);
    uint8_t TR_P4 = wb_transitions_around(d_src, row, col - 1, width);
    uint8_t P2 = P2(d_src, row, col, width);
    uint8_t P4 = P4(d_src, row, col, width);
    uint8_t P6 = P6(d_src, row, col, width);
    uint8_t P8 = P8(d_src, row, col, width);

    uint8_t thinning_cond_1 = ((2 <= NZ) & (NZ <= 6));
    uint8_t thinning_cond_2 = (TR_P1 == 1);
    uint8_t thinning_cond_3 = (((P2 & P4 & P8) == 0) | (TR_P2 != 1));
    uint8_t thinning_cond_4 = (((P2 & P4 & P6) == 0) | (TR_P4 != 1));
    uint8_t thinning_cond_ok = thinning_cond_1 & thinning_cond_2 & thinning_cond_3 & thinning_cond_4;

    d_dst[row * width + col] = BINARY_WHITE + ((1 - thinning_cond_ok) * d_src[row * width + col]);
}

// Computes the number of white to black transitions around a pixel.
__device__ uint8_t wb_transitions_around(uint8_t* d_data, unsigned int row, unsigned int col, unsigned int width) {
    uint8_t count = 0;

    count += ( (P2(d_data, row, col, width) == BINARY_WHITE) & (P3(d_data, row, col, width) == BINARY_BLACK) );
    count += ( (P3(d_data, row, col, width) == BINARY_WHITE) & (P4(d_data, row, col, width) == BINARY_BLACK) );
    count += ( (P4(d_data, row, col, width) == BINARY_WHITE) & (P5(d_data, row, col, width) == BINARY_BLACK) );
    count += ( (P5(d_data, row, col, width) == BINARY_WHITE) & (P6(d_data, row, col, width) == BINARY_BLACK) );
    count += ( (P6(d_data, row, col, width) == BINARY_WHITE) & (P7(d_data, row, col, width) == BINARY_BLACK) );
    count += ( (P7(d_data, row, col, width) == BINARY_WHITE) & (P8(d_data, row, col, width) == BINARY_BLACK) );
    count += ( (P8(d_data, row, col, width) == BINARY_WHITE) & (P9(d_data, row, col, width) == BINARY_BLACK) );
    count += ( (P9(d_data, row, col, width) == BINARY_WHITE) & (P2(d_data, row, col, width) == BINARY_BLACK) );

    return count;
}

int main(int argc, char** argv) {
    Bitmap* src_bitmap = NULL;
    Bitmap* dst_bitmap = NULL;
    Padding padding;
    dim3 grid_dim;
    dim3 block_dim;

    gpu_pre_skeletonization(argc, argv, &src_bitmap, &dst_bitmap, &padding, &grid_dim, &block_dim);

    unsigned int iterations = skeletonize(&src_bitmap, &dst_bitmap, padding, grid_dim, block_dim);
    printf(" %u iterations\n", iterations);

    gpu_post_skeletonization(argv, &src_bitmap, &dst_bitmap, &padding);

    return EXIT_SUCCESS;
}
