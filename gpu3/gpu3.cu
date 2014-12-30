#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu3.cuh"
#include "../common/gpu_only_utils.cuh"
#include "../common/lspbmp.hpp"
#include "../common/utils.hpp"

#define PAD_TOP 2
#define PAD_LEFT 2
#define PAD_BOTTOM 1
#define PAD_RIGHT 1

void and_reduction(uint8_t* d_data, int width, int height, dim3 grid_dim, dim3 block_dim) {
    int shared_mem_size = block_dim.x * block_dim.y * sizeof(uint8_t);

    // iterative reductions of d_data
    do {
        and_reduction<<<grid_dim, block_dim, shared_mem_size>>>(d_data, width, height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        width = grid_dim.x;
        height = grid_dim.y;
        grid_dim.x = ceil(grid_dim.x / ((double) block_dim.x));
        grid_dim.y = ceil(grid_dim.y / ((double) block_dim.y));
    } while ((width * height) != 1);
}

// Adapted for 2D arrays from Nvidia cuda SDK samples
__global__ void and_reduction(uint8_t* d_data, int width, int height) {
    // shared memory for tile
    extern __shared__ uint8_t s_data[];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Load equality values into shared memory tile. We use 1 as the default
    // value, as it is an AND reduction
    s_data[tid] = is_outside_image(row, col, width, height) ? 1 : global_mem_read(d_data, row, col, width, height);;
    __syncthreads();

    // do reduction in shared memory
    for (int s = ((blockDim.x * blockDim.y) / 2); s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] &= s_data[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global memory
    if (tid == 0) {
        int write_data = s_data[0];
        global_mem_write(d_data, blockIdx.y, blockIdx.x, gridDim.x, gridDim.y, write_data);
    }
}

// Computes the number of black neighbors around a pixel.
__device__ uint8_t black_neighbors_around(uint8_t* d_data, int row, int col, int width) {
    uint8_t count = 0;

    count += (P2_f(d_data, row, col, width) == BINARY_BLACK);
    count += (P3_f(d_data, row, col, width) == BINARY_BLACK);
    count += (P4_f(d_data, row, col, width) == BINARY_BLACK);
    count += (P5_f(d_data, row, col, width) == BINARY_BLACK);
    count += (P6_f(d_data, row, col, width) == BINARY_BLACK);
    count += (P7_f(d_data, row, col, width) == BINARY_BLACK);
    count += (P8_f(d_data, row, col, width) == BINARY_BLACK);
    count += (P9_f(d_data, row, col, width) == BINARY_BLACK);

    return count;
}

__device__ uint8_t global_mem_read(uint8_t* d_data, int row, int col, int width, int height) {
    return is_outside_image(row, col, width, height) ? BINARY_WHITE : d_data[row * width + col];
}

__device__ void global_mem_write(uint8_t* d_data, int row, int col, int width, int height, uint8_t write_data) {
    if (!is_outside_image(row, col, width, height)) {
        d_data[row * width + col] = write_data;
    }
}

__device__ uint8_t is_outside_image(int row, int col, int width, int height) {
    return (row < 0) | (row > (height - 1)) | (col < 0) | (col > (width - 1));
}

__device__ uint8_t P2_f(uint8_t* data, int row, int col, int width) {
    return data[(row - 1) * width + col];
}

__device__ uint8_t P3_f(uint8_t* data, int row, int col, int width) {
    return data[(row - 1) * width + (col - 1)];
}

__device__ uint8_t P4_f(uint8_t* data, int row, int col, int width) {
    return data[row * width + (col - 1)];
}

__device__ uint8_t P5_f(uint8_t* data, int row, int col, int width) {
    return data[(row + 1) * width + (col - 1)];
}

__device__ uint8_t P6_f(uint8_t* data, int row, int col, int width) {
    return data[(row + 1) * width + col];
}

__device__ uint8_t P7_f(uint8_t* data, int row, int col, int width) {
    return data[(row + 1) * width + (col + 1)];
}

__device__ uint8_t P8_f(uint8_t* data, int row, int col, int width) {
    return data[row * width + (col + 1)];
}

__device__ uint8_t P9_f(uint8_t* data, int row, int col, int width) {
    return data[(row - 1) * width + (col + 1)];
}

__global__ void pixel_equality(uint8_t* d_in_1, uint8_t* d_in_2, uint8_t* d_out, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int write_data = (global_mem_read(d_in_1, row, col, width, height) == global_mem_read(d_in_2, row, col, width, height));
    global_mem_write(d_out, row, col, width, height, write_data);
}

// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3 grid_dim, dim3 block_dim) {
    // allocate memory on device
    uint8_t* d_src_data = NULL;
    uint8_t* d_dst_data = NULL;
    uint8_t* d_equ_data = NULL;
    int data_size = (*src_bitmap)->width * (*src_bitmap)->height * sizeof(uint8_t);
    gpuErrchk(cudaMalloc((void**) &d_src_data, data_size));
    gpuErrchk(cudaMalloc((void**) &d_dst_data, data_size));
    gpuErrchk(cudaMalloc((void**) &d_equ_data, data_size));

    // send data to device
    gpuErrchk(cudaMemcpy(d_src_data, (*src_bitmap)->data, data_size, cudaMemcpyHostToDevice));

    uint8_t are_identical_bitmaps = 0;
    int iterations = 0;
    do {
        int skeletonize_pass_shared_mem_size = (block_dim.x + PAD_LEFT + PAD_RIGHT) * (block_dim.y + PAD_TOP + PAD_BOTTOM) * sizeof(uint8_t);
        skeletonize_pass<<<grid_dim, block_dim, skeletonize_pass_shared_mem_size>>>(d_src_data, d_dst_data, (*src_bitmap)->width, (*src_bitmap)->height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        pixel_equality<<<grid_dim, block_dim>>>(d_src_data, d_dst_data, d_equ_data, (*src_bitmap)->width, (*src_bitmap)->height);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        and_reduction(d_equ_data, (*src_bitmap)->width, (*src_bitmap)->height, grid_dim, block_dim);

        // bring reduced bitmap equality information back from device
        gpuErrchk(cudaMemcpy(&are_identical_bitmaps, d_equ_data, 1 * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        swap_bitmaps((void**) &d_src_data, (void**) &d_dst_data);

        iterations++;
        printf(".");
        fflush(stdout);
    } while (!are_identical_bitmaps);

    // bring dst_bitmap back from device
    gpuErrchk(cudaMemcpy((*dst_bitmap)->data, d_dst_data, data_size, cudaMemcpyDeviceToHost));

    // free memory on device
    gpuErrchk(cudaFree(d_src_data));
    gpuErrchk(cudaFree(d_dst_data));
    gpuErrchk(cudaFree(d_equ_data));

    return iterations;
}

// Performs 1 iteration of the thinning algorithm.
__global__ void skeletonize_pass(uint8_t* d_src, uint8_t* d_dst, int d_width, int d_height) {
    // shared memory for tile
    extern __shared__ uint8_t s_src[];

    int d_row = blockIdx.y * blockDim.y + threadIdx.y;
    int d_col = blockIdx.x * blockDim.x + threadIdx.x;

    int s_row = threadIdx.y + PAD_TOP;
    int s_col = threadIdx.x + PAD_LEFT;
    int s_width = blockDim.x + PAD_LEFT + PAD_RIGHT;

    // load data into shared memory
    if ((threadIdx.y == 0) & (threadIdx.x == 0)) {
        // top-left corner
        s_src[(s_row - 2) * s_width + (s_col - 2)] = global_mem_read(d_src, d_row - 2, d_col - 2, d_width, d_height);
        s_src[(s_row - 2) * s_width + (s_col - 1)] = global_mem_read(d_src, d_row - 2, d_col - 1, d_width, d_height);
        s_src[(s_row - 2) * s_width + (s_col)] = global_mem_read(d_src, d_row - 2, d_col, d_width, d_height);

        s_src[(s_row - 1) * s_width + (s_col - 2)] = global_mem_read(d_src, d_row - 1, d_col - 2, d_width, d_height);
        s_src[(s_row - 1) * s_width + (s_col - 1)] = global_mem_read(d_src, d_row - 1, d_col - 1, d_width, d_height);
        s_src[(s_row - 1) * s_width + (s_col)] = global_mem_read(d_src, d_row - 1, d_col, d_width, d_height);

        s_src[(s_row) * s_width + (s_col - 2)] = global_mem_read(d_src, d_row, d_col - 2, d_width, d_height);
        s_src[(s_row) * s_width + (s_col - 1)] = global_mem_read(d_src, d_row, d_col - 1, d_width, d_height);
        s_src[(s_row) * s_width + (s_col)] = global_mem_read(d_src, d_row, d_col, d_width, d_height);
    } else if ((threadIdx.y == (blockDim.y - 1)) & (threadIdx.x == 0)) {
        // bottom-left corner
        s_src[(s_row) * s_width + (s_col - 2)] = global_mem_read(d_src, d_row, d_col - 2, d_width, d_height);
        s_src[(s_row) * s_width + (s_col - 1)] = global_mem_read(d_src, d_row, d_col - 1, d_width, d_height);
        s_src[(s_row) * s_width + (s_col)] = global_mem_read(d_src, d_row, d_col, d_width, d_height);

        s_src[(s_row + 1) * s_width + (s_col - 2)] = global_mem_read(d_src, d_row + 1, d_col - 2, d_width, d_height);
        s_src[(s_row + 1) * s_width + (s_col - 1)] = global_mem_read(d_src, d_row + 1, d_col - 1, d_width, d_height);
        s_src[(s_row + 1) * s_width + (s_col)] = global_mem_read(d_src, d_row + 1, d_col, d_width, d_height);
    } else if ((threadIdx.y == (blockDim.y - 1)) & (threadIdx.x == (blockDim.x - 1))) {
        // bottom-right corner
        s_src[(s_row) * s_width + (s_col)] = global_mem_read(d_src, d_row, d_col, d_width, d_height);
        s_src[(s_row) * s_width + (s_col + 1)] = global_mem_read(d_src, d_row, d_col + 1, d_width, d_height);

        s_src[(s_row + 1) * s_width + (s_col)] = global_mem_read(d_src, d_row + 1, d_col, d_width, d_height);
        s_src[(s_row + 1) * s_width + (s_col + 1)] = global_mem_read(d_src, d_row + 1, d_col + 1, d_width, d_height);
    } else if ((threadIdx.y == 0) & (threadIdx.x == (blockDim.x - 1))) {
        // top-right corner
        s_src[(s_row - 2) * s_width + (s_col)] = global_mem_read(d_src, d_row - 2, d_col, d_width, d_height);
        s_src[(s_row - 2) * s_width + (s_col + 1)] = global_mem_read(d_src, d_row - 2, d_col + 1, d_width, d_height);

        s_src[(s_row - 1) * s_width + (s_col)] = global_mem_read(d_src, d_row - 1, d_col, d_width, d_height);
        s_src[(s_row - 1) * s_width + (s_col + 1)] = global_mem_read(d_src, d_row - 1, d_col + 1, d_width, d_height);

        s_src[(s_row) * s_width + (s_col)] = global_mem_read(d_src, d_row, d_col, d_width, d_height);
        s_src[(s_row) * s_width + (s_col + 1)] = global_mem_read(d_src, d_row, d_col + 1, d_width, d_height);
    } else if (threadIdx.y == 0) {
        // PAD_TOP top rows
        s_src[(s_row - 2) * s_width + (s_col)] = global_mem_read(d_src, d_row - 2, d_col, d_width, d_height);
        s_src[(s_row - 1) * s_width + (s_col)] = global_mem_read(d_src, d_row - 1, d_col, d_width, d_height);
        s_src[(s_row) * s_width + (s_col)] = global_mem_read(d_src, d_row, d_col, d_width, d_height);
    } else if (threadIdx.x == 0) {
        // PAD_LEFT left rows
        s_src[(s_row) * s_width + (s_col - 2)] = global_mem_read(d_src, d_row, d_col - 2, d_width, d_height);
        s_src[(s_row) * s_width + (s_col - 1)] = global_mem_read(d_src, d_row, d_col - 1, d_width, d_height);
        s_src[(s_row) * s_width + (s_col)] = global_mem_read(d_src, d_row, d_col, d_width, d_height);
    } else if (threadIdx.y == (blockDim.y - 1)) {
        // PAD_BOTTOM bottom rows
        s_src[(s_row) * s_width + (s_col)] = global_mem_read(d_src, d_row, d_col, d_width, d_height);
        s_src[(s_row + 1) * s_width + (s_col)] = global_mem_read(d_src, d_row + 1, d_col, d_width, d_height);
    } else if (threadIdx.x == (blockDim.x - 1)) {
        // PAD_RIGHT right rows
        s_src[(s_row) * s_width + (s_col)] = global_mem_read(d_src, d_row, d_col, d_width, d_height);
        s_src[(s_row) * s_width + (s_col + 1)] = global_mem_read(d_src, d_row, d_col + 1, d_width, d_height);
    } else {
        // center pixels
        s_src[(s_row) * s_width + (s_col)] = global_mem_read(d_src, d_row, d_col, d_width, d_height);
    }

    __syncthreads();

    uint8_t NZ = black_neighbors_around(s_src, s_row, s_col, s_width);
    uint8_t TR_P1 = wb_transitions_around(s_src, s_row, s_col, s_width);
    uint8_t TR_P2 = wb_transitions_around(s_src, s_row - 1, s_col, s_width);
    uint8_t TR_P4 = wb_transitions_around(s_src, s_row, s_col - 1, s_width);
    uint8_t P2 = P2_f(s_src, s_row, s_col, s_width);
    uint8_t P4 = P4_f(s_src, s_row, s_col, s_width);
    uint8_t P6 = P6_f(s_src, s_row, s_col, s_width);
    uint8_t P8 = P8_f(s_src, s_row, s_col, s_width);

    uint8_t thinning_cond_1 = ((2 <= NZ) & (NZ <= 6));
    uint8_t thinning_cond_2 = (TR_P1 == 1);
    uint8_t thinning_cond_3 = (((P2 & P4 & P8) == 0) | (TR_P2 != 1));
    uint8_t thinning_cond_4 = (((P2 & P4 & P6) == 0) | (TR_P4 != 1));
    uint8_t thinning_cond_ok = thinning_cond_1 & thinning_cond_2 & thinning_cond_3 & thinning_cond_4;

    uint8_t write_data = BINARY_WHITE + ((1 - thinning_cond_ok) * s_src[s_row * s_width + s_col]);
    global_mem_write(d_dst, d_row, d_col, d_width, d_height, write_data);
}

// Computes the number of white to black transitions around a pixel.
__device__ uint8_t wb_transitions_around(uint8_t* d_data, int row, int col, int width) {
    uint8_t count = 0;

    count += ((P2_f(d_data, row, col, width) == BINARY_WHITE) & (P3_f(d_data, row, col, width) == BINARY_BLACK));
    count += ((P3_f(d_data, row, col, width) == BINARY_WHITE) & (P4_f(d_data, row, col, width) == BINARY_BLACK));
    count += ((P4_f(d_data, row, col, width) == BINARY_WHITE) & (P5_f(d_data, row, col, width) == BINARY_BLACK));
    count += ((P5_f(d_data, row, col, width) == BINARY_WHITE) & (P6_f(d_data, row, col, width) == BINARY_BLACK));
    count += ((P6_f(d_data, row, col, width) == BINARY_WHITE) & (P7_f(d_data, row, col, width) == BINARY_BLACK));
    count += ((P7_f(d_data, row, col, width) == BINARY_WHITE) & (P8_f(d_data, row, col, width) == BINARY_BLACK));
    count += ((P8_f(d_data, row, col, width) == BINARY_WHITE) & (P9_f(d_data, row, col, width) == BINARY_BLACK));
    count += ((P9_f(d_data, row, col, width) == BINARY_WHITE) & (P2_f(d_data, row, col, width) == BINARY_BLACK));

    return count;
}

int main(int argc, char** argv) {
    Bitmap* src_bitmap = NULL;
    Bitmap* dst_bitmap = NULL;
    dim3 grid_dim;
    dim3 block_dim;

    gpu_pre_skeletonization(argc, argv, &src_bitmap, &dst_bitmap, &grid_dim, &block_dim);

    int iterations = skeletonize(&src_bitmap, &dst_bitmap, grid_dim, block_dim);
    printf(" %u iterations\n", iterations);
    printf("\n");

    gpu_post_skeletonization(argv, &src_bitmap, &dst_bitmap);

    return EXIT_SUCCESS;
}
