#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu1.cuh"
#include "../common/utils.hpp"

#define MAX_THREADS_PER_BLOCK 1024

#define PAD_TOP 2
#define PAD_LEFT 2
#define PAD_BOTTOM 1
#define PAD_RIGHT 1

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

    // for the dst data, we don't need to actually send the real data. All we
    // need is to send some data that is correctly padded with BINARY_WHITE on
    // the sides.
    cudaMemset(d_dst_data, BINARY_WHITE, data_size);

    unsigned int iterations = 0;
    do {
        skeletonize_pass<<<grid_dim, block_dim>>>(d_src_data, d_dst_data, (*src_bitmap)->width, padding);

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
    assert(argc == 5 && "Usage: gpu1 <input_file_name.bmp> <output_file_name.bmp> <block_dim_x> <block_dim_y>");

    char* src_fname = argv[1];
    char* dst_fname = argv[2];
    char* block_dim_x_string = argv[3];
    char* block_dim_y_string = argv[4];

    printf("src_fname   = %s\n", src_fname);
    printf("dst_fname   = %s\n", dst_fname);

    // load src image
    Bitmap* src_bitmap = loadBitmap(src_fname);
    assert(src_bitmap != NULL && "Error: could not load src bitmap");

    // validate src image is 8-bit binary-valued grayscale image
    assert(is_binary_valued_grayscale_image(src_bitmap) && "Error: Only 8-bit binary-valued grayscale images are supported. Values must be black (0) or white (255) only");

    // we work on true binary images
    grayscale_to_binary(src_bitmap);

    // Create dst bitmap image (empty for now)
    Bitmap* dst_bitmap = createBitmap(src_bitmap->width, src_bitmap->height, src_bitmap->depth);
    assert(dst_bitmap != NULL && "Error: could not allocate memory for dst bitmap");

    // Dimensions of computing elements on the CUDA device.
    // Computing the grid dimensions depends on PAD_TOP and PAD_LEFT.
    unsigned int block_dim_x = strtol(block_dim_x_string, NULL, 10);
    unsigned int block_dim_y = strtol(block_dim_y_string, NULL, 10);
    assert((block_dim_x * block_dim_y) <= MAX_THREADS_PER_BLOCK);

    unsigned int grid_dim_x = (unsigned int) ceil((src_bitmap->width) / ((double) block_dim_x));
    unsigned int grid_dim_y = (unsigned int) ceil((src_bitmap->height)/ ((double) block_dim_y));
    dim3 block_dim(block_dim_x, block_dim_y);
    dim3 grid_dim(grid_dim_x, grid_dim_y);

    printf("orig img width = %u\n", src_bitmap->width);
    printf("orig img height = %u\n", src_bitmap->height);

    // Pad the binary images with pixels on each side. This will be useful when
    // implementing the skeletonization algorithm, because the mask we use
    // depends on P2 and P4, which also have their own window.
    // ATTENTION : it is important to use cast to (int) since we want to test
    // for a maximum value and the subtraction can yield a negative number.
    Padding padding;
    padding.top = PAD_TOP;
    padding.bottom = max((int) ((grid_dim_y * block_dim_y) - (src_bitmap->height + PAD_BOTTOM)), PAD_BOTTOM);
    padding.left = PAD_LEFT;
    padding.right = max((int) ((grid_dim_x * block_dim_x) - (src_bitmap->width + PAD_RIGHT)), PAD_RIGHT);
    pad_binary_bitmap(&src_bitmap, BINARY_WHITE, padding);
    pad_binary_bitmap(&dst_bitmap, BINARY_WHITE, padding);

    printf("padded img width = %u\n", src_bitmap->width);
    printf("padded img height = %u\n", src_bitmap->height);
    printf("block dim X = %u\n", block_dim_x);
    printf("block dim Y = %u\n", block_dim_y);
    printf("grid dim X = %u\n", grid_dim_x);
    printf("grid dim Y = %u\n", grid_dim_y);

    unsigned int iterations = skeletonize(&src_bitmap, &dst_bitmap, padding, grid_dim, block_dim);
    printf(" %u iterations\n", iterations);

    // Remove extra padding that was added to the images (don't care about
    // src_bitmap, so only need to unpad dst_bitmap)
    unpad_binary_bitmap(&dst_bitmap, padding);

    // save 8-bit binary-valued grayscale version of dst_bitmap to dst_fname
    binary_to_grayscale(dst_bitmap);
    int save_successful = saveBitmap(dst_fname, dst_bitmap);
    assert(save_successful == 1 && "Error: could not save dst bitmap");

    // free memory used for bitmaps
    free(src_bitmap);
    free(dst_bitmap);

    return EXIT_SUCCESS;
}
