#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu1.cuh"
#include "../common/gpu_only_utils.cuh"
#include "../common/utils.hpp"

#define P2(d_data, row, col, width, height) (is_outside_image((row) - 1, (col), (width), (height)) ? BINARY_WHITE : (d_data)[((row) - 1) * (width) + (col)])
#define P3(d_data, row, col, width, height) (is_outside_image((row) - 1, (col) - 1, (width), (height)) ? BINARY_WHITE : (d_data)[((row) - 1) * (width) + ((col) - 1)])
#define P4(d_data, row, col, width, height) (is_outside_image((row), (col) - 1, (width), (height)) ? BINARY_WHITE : (d_data)[(row) * (width) + ((col) - 1)])
#define P5(d_data, row, col, width, height) (is_outside_image((row) + 1, (col) - 1, (width), (height)) ? BINARY_WHITE : (d_data)[((row) + 1) * (width) + ((col) - 1)])
#define P6(d_data, row, col, width, height) (is_outside_image((row) + 1, (col), (width), (height)) ? BINARY_WHITE : (d_data)[((row) + 1) * (width) + (col)])
#define P7(d_data, row, col, width, height) (is_outside_image((row) + 1, (col) + 1, (width), (height)) ? BINARY_WHITE : (d_data)[((row) + 1) * (width) + ((col) + 1)])
#define P8(d_data, row, col, width, height) (is_outside_image((row), (col) + 1, (width), (height)) ? BINARY_WHITE : (d_data)[(row) * (width) + ((col) + 1)])
#define P9(d_data, row, col, width, height) (is_outside_image((row) - 1, (col) + 1, (width), (height)) ? BINARY_WHITE : (d_data)[((row) - 1) * (width) + ((col) + 1)])

// Computes the number of black neighbors around a pixel.
__device__ uint8_t black_neighbors_around(uint8_t* d_data, int row, int col, unsigned int width, unsigned int height, unsigned int iterations) {
    uint8_t count = 0;

    // if (row == 1348 && col == 777 && iterations == 0) {
    //     printf("P2 = %u\n", P2(d_data, row, col, width, height));
    //     printf("P3 = %u\n", P3(d_data, row, col, width, height));
    //     printf("P4 = %u\n", P4(d_data, row, col, width, height));
    //     printf("P5 = %u\n", P5(d_data, row, col, width, height));
    //     printf("P6 = %u\n", P6(d_data, row, col, width, height));
    //     printf("P7 = %u\n", P7(d_data, row, col, width, height));
    //     printf("P8 = %u\n", P8(d_data, row, col, width, height));
    //     printf("P9 = %u\n", P9(d_data, row, col, width, height));
    // }

    count += (P2(d_data, row, col, width, height) == BINARY_BLACK);
    count += (P3(d_data, row, col, width, height) == BINARY_BLACK);
    count += (P4(d_data, row, col, width, height) == BINARY_BLACK);
    count += (P5(d_data, row, col, width, height) == BINARY_BLACK);
    count += (P6(d_data, row, col, width, height) == BINARY_BLACK);
    count += (P7(d_data, row, col, width, height) == BINARY_BLACK);
    count += (P8(d_data, row, col, width, height) == BINARY_BLACK);
    count += (P9(d_data, row, col, width, height) == BINARY_BLACK);

    return count;
}

// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3 grid_dim, dim3 block_dim) {
    // allocate memory on device
    uint8_t* d_src_data = NULL;
    uint8_t* d_dst_data = NULL;
    unsigned int data_size = (*src_bitmap)->width * (*src_bitmap)->height * sizeof(uint8_t);
    gpuErrchk(cudaMalloc((void**) &d_src_data, data_size));
    gpuErrchk(cudaMalloc((void**) &d_dst_data, data_size));

    // send data to device
    gpuErrchk(cudaMemcpy(d_src_data, (*src_bitmap)->data, data_size, cudaMemcpyHostToDevice));

    unsigned int iterations = 0;
    do {
        skeletonize_pass<<<grid_dim, block_dim>>>(d_src_data, d_dst_data, (*src_bitmap)->width, (*src_bitmap)->height, iterations);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // bring data back from device
        gpuErrchk(cudaMemcpy((*src_bitmap)->data, d_src_data, data_size, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy((*dst_bitmap)->data, d_dst_data, data_size, cudaMemcpyDeviceToHost));

        swap_bitmaps((void**) &d_src_data, (void**) &d_dst_data);

        iterations++;
        printf(".");
        fflush(stdout);
    } while (!are_identical_bitmaps(*src_bitmap, *dst_bitmap));

    // free memory on device
    gpuErrchk(cudaFree(d_src_data));
    gpuErrchk(cudaFree(d_dst_data));

    return iterations;
}

// Performs 1 iteration of the thinning algorithm.
__global__ void skeletonize_pass(uint8_t* d_src, uint8_t* d_dst, unsigned int width, unsigned int height, unsigned int iterations) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t NZ = black_neighbors_around(d_src, row, col, width, height, iterations);
    uint8_t TR_P1 = wb_transitions_around(d_src, row, col, width, height, iterations);
    uint8_t TR_P2 = wb_transitions_around(d_src, row - 1, col, width, height, iterations);
    uint8_t TR_P4 = wb_transitions_around(d_src, row, col - 1, width, height, iterations);
    uint8_t P2 = P2(d_src, row, col, width, height);
    uint8_t P4 = P4(d_src, row, col, width, height);
    uint8_t P6 = P6(d_src, row, col, width, height);
    uint8_t P8 = P8(d_src, row, col, width, height);

    uint8_t thinning_cond_1 = ((2 <= NZ) & (NZ <= 6));
    uint8_t thinning_cond_2 = (TR_P1 == 1);
    uint8_t thinning_cond_3 = (((P2 & P4 & P8) == 0) | (TR_P2 != 1));
    uint8_t thinning_cond_4 = (((P2 & P4 & P6) == 0) | (TR_P4 != 1));
    uint8_t thinning_cond_ok = thinning_cond_1 & thinning_cond_2 & thinning_cond_3 & thinning_cond_4;

    // if (row == 1348 && col == 777 && iterations == 0) {
    //     printf("======================\n");
    //     printf("src[%u * %u + %u] = %u\n", row, width, col, d_src[row * width + col]);
    //     printf("NZ = %u\n", NZ);
    //     printf("TR_P1 = %u\n", TR_P1);
    //     printf("TR_P2 = %u\n", TR_P2);
    //     printf("TR_P4 = %u\n", TR_P4);
    //     printf("P2 = %u\n", P2);
    //     printf("P4 = %u\n", P4);
    //     printf("P6 = %u\n", P6);
    //     printf("P8 = %u\n", P8);
    //     printf("thinning_cond_1 = %u\n", thinning_cond_1);
    //     printf("thinning_cond_2 = %u\n", thinning_cond_2);
    //     printf("thinning_cond_3 = %u\n", thinning_cond_3);
    //     printf("thinning_cond_4 = %u\n", thinning_cond_4);
    //     printf("======================\n");
    // }

    d_dst[row * width + col] = BINARY_WHITE + ((1 - thinning_cond_ok) * d_src[row * width + col]);
}

// Computes the number of white to black transitions around a pixel.
__device__ uint8_t wb_transitions_around(uint8_t* d_data, int row, int col, unsigned int width, unsigned int height, unsigned int iterations) {
    uint8_t count = 0;

    count += ((P2(d_data, row, col, width, height) == BINARY_WHITE) & (P3(d_data, row, col, width, height) == BINARY_BLACK));
    count += ((P3(d_data, row, col, width, height) == BINARY_WHITE) & (P4(d_data, row, col, width, height) == BINARY_BLACK));
    count += ((P4(d_data, row, col, width, height) == BINARY_WHITE) & (P5(d_data, row, col, width, height) == BINARY_BLACK));
    count += ((P5(d_data, row, col, width, height) == BINARY_WHITE) & (P6(d_data, row, col, width, height) == BINARY_BLACK));
    count += ((P6(d_data, row, col, width, height) == BINARY_WHITE) & (P7(d_data, row, col, width, height) == BINARY_BLACK));
    count += ((P7(d_data, row, col, width, height) == BINARY_WHITE) & (P8(d_data, row, col, width, height) == BINARY_BLACK));
    count += ((P8(d_data, row, col, width, height) == BINARY_WHITE) & (P9(d_data, row, col, width, height) == BINARY_BLACK));
    count += ((P9(d_data, row, col, width, height) == BINARY_WHITE) & (P2(d_data, row, col, width, height) == BINARY_BLACK));

    return count;
}

int main(int argc, char** argv) {
    Bitmap* src_bitmap = NULL;
    Bitmap* dst_bitmap = NULL;
    Padding padding;
    dim3 grid_dim;
    dim3 block_dim;

    gpu_pre_skeletonization(argc, argv, &src_bitmap, &dst_bitmap, &padding, &grid_dim, &block_dim);

    unsigned int iterations = skeletonize(&src_bitmap, &dst_bitmap, grid_dim, block_dim);
    printf(" %u iterations\n", iterations);

    gpu_post_skeletonization(argv, &src_bitmap, &dst_bitmap, &padding);

    return EXIT_SUCCESS;
}
