#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu2.cuh"
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

void and_reduction(dim3 grid_dim, dim3 block_dim, uint8_t* d_pixel_equ, unsigned int pixel_equ_size) {
    unsigned int shared_mem_size = block_dim.x * block_dim.y * sizeof(uint8_t);
    unsigned int grid_size = grid_dim.x * grid_dim.y;
    unsigned int block_size = block_dim.x * block_dim.y;

    // iterative reductions of d_pixel_equ
    do {
        and_reduction<<<grid_size, block_size, shared_mem_size>>>(d_pixel_equ, pixel_equ_size);
        pixel_equ_size = grid_size;
        grid_size = ceil(grid_size / ((double) block_size));
    } while (pixel_equ_size != 1);
}

// Adapted from Nvidia cuda SDK samples
__global__ void and_reduction(uint8_t* d_data, unsigned int size) {
    // shared memory for tile (without padding, unlike in skeletonize_pass)
    extern __shared__ uint8_t s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // load equality values into shared memory tile
    s_data[tid] = (i < size) ? d_data[i] : 1; // we use 1 since it is a binary AND
    __syncthreads();

    // do reduction in shared memory
    for (unsigned int s = (blockDim.x / 2); s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] &= s_data[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0) {
        d_data[blockIdx.x] = s_data[0];
    }
}

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

__global__ void pixel_equality(uint8_t* d_in_1, uint8_t* d_in_2, uint8_t* d_out, unsigned int width, Padding padding) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y + padding.top;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x + padding.left;

    d_out[(row - padding.top) * (width - padding.left - padding.right) + (col - padding.left)] = (d_in_1[row * width + col] == d_in_2[row * width + col]);
}

// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding padding, dim3 grid_dim, dim3 block_dim) {
    // 1 byte of data indicating if the images are identical
    uint8_t grid_equ = 0;

    // allocate memory on device
    uint8_t* d_src_data = NULL;
    uint8_t* d_dst_data = NULL;
    uint8_t* d_pixel_equ = NULL;
    uint8_t* d_block_equ = NULL;
    uint8_t* d_grid_equ = NULL;

    unsigned int data_size = (*src_bitmap)->width * (*src_bitmap)->height * sizeof(uint8_t);
    unsigned int pixel_equ_size = ((*src_bitmap)->width - padding.left - padding.right) * ((*src_bitmap)->height - padding.top - padding.bottom) * sizeof(uint8_t);
    unsigned int block_equ_size = grid_dim.x * grid_dim.y * sizeof(uint8_t);
    unsigned int grid_equ_size = 1 * sizeof(uint8_t);

    cudaError d_src_malloc_success = cudaMalloc((void**) &d_src_data, data_size);
    cudaError d_dst_malloc_success = cudaMalloc((void**) &d_dst_data, data_size);
    cudaError d_pixel_equ_malloc_success = cudaMalloc((void**) &d_pixel_equ, pixel_equ_size);
    cudaError d_block_equ_malloc_success = cudaMalloc((void**) &d_block_equ, block_equ_size);
    cudaError d_grid_equ_malloc_success = cudaMalloc((void**) &d_grid_equ, grid_equ_size);

    assert((d_src_malloc_success == cudaSuccess) && "Error: could not allocate memory for d_src_data");
    assert((d_dst_malloc_success == cudaSuccess) && "Error: could not allocate memory for d_dst_data");
    assert((d_pixel_equ_malloc_success == cudaSuccess) && "Error: could not allocate memory for d_pixel_equ");
    assert((d_block_equ_malloc_success == cudaSuccess) && "Error: could not allocate memory for d_block_equ");
    assert((d_grid_equ_malloc_success == cudaSuccess) && "Error: could not allocate memory for d_grid_equ");

    // send data to device
    cudaMemcpy(d_src_data, (*src_bitmap)->data, data_size, cudaMemcpyHostToDevice);

    // for dst_data, we don't need to actually send the real data. All we need
    // is to send some data that is correctly padded with BINARY_WHITE on the
    // sides.
    cudaMemset(d_dst_data, BINARY_WHITE, data_size);

    unsigned int iterations = 0;
    do {
        // 2D grid & 2D block
        skeletonize_pass<<<grid_dim, block_dim>>>(d_src_data, d_dst_data, (*src_bitmap)->width, padding);
        pixel_equality<<<grid_dim, block_dim>>>(d_src_data, d_dst_data, d_pixel_equ, (*src_bitmap)->width, padding);

        // 1D grid & 1D block reduction
        and_reduction(grid_dim, block_dim, d_pixel_equ, pixel_equ_size);

        // bring d_grid_equ back from device
        cudaMemcpy(&grid_equ, d_pixel_equ, grid_equ_size, cudaMemcpyDeviceToHost);

        swap_bitmaps((void**) &d_src_data, (void**) &d_dst_data);

        iterations++;
        printf(".");
        fflush(stdout);
    } while (!grid_equ);

    // bring data back from device
    cudaMemcpy((*dst_bitmap)->data, d_dst_data, data_size, cudaMemcpyDeviceToHost);

    // free memory on device
    cudaFree(d_src_data);
    cudaFree(d_dst_data);
    cudaFree(d_pixel_equ);
    cudaFree(d_block_equ);
    cudaFree(d_grid_equ);

    return iterations;
}

// Performs 1 iteration of the thinning algorithm.
__global__ void skeletonize_pass(uint8_t* d_src, uint8_t* d_dst, unsigned int width, Padding padding) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y + padding.top;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x + padding.left;

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
