#include <assert.h>
#include <libgen.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu1.cuh"
#include "../common/files.hpp"
#include "../common/utils.hpp"

#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

#define PAD_TOP 2
#define PAD_LEFT 2
#define PAD_BOTTOM 1
#define PAD_RIGHT 1

#define P2(d_data, row, col, width) ((d_data)[((row)-1) * width +  (col)   ])
#define P3(d_data, row, col, width) ((d_data)[((row)-1) * width + ((col)-1)])
#define P4(d_data, row, col, width) ((d_data)[ (row)    * width + ((col)-1)])
#define P5(d_data, row, col, width) ((d_data)[((row)+1) * width + ((col)-1)])
#define P6(d_data, row, col, width) ((d_data)[((row)+1) * width +  (col)   ])
#define P7(d_data, row, col, width) ((d_data)[((row)+1) * width + ((col)+1)])
#define P8(d_data, row, col, width) ((d_data)[ (row)    * width + ((col)+1)])
#define P9(d_data, row, col, width) ((d_data)[((row)-1) * width + ((col)+1)])

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

// Performs an image skeletonization algorithm on the input file, and stores the
// result in the specified output file.
unsigned int skeletonize(const char* src_fname, const char* dst_fname) {
    assert(src_fname && "src_fname must be non-NULL");
    assert(dst_fname && "dst_fname must be non-NULL");

    // load src image
    Bitmap* src_bitmap = loadBitmap(src_fname);
    assert(src_bitmap && "Bitmap has to be non-NULL");

    // validate src image is 8-bit binary-valued grayscale image
    assert(is_binary_valued_grayscale_image(src_bitmap) && "Only 8-bit binary-valued grayscale images are supported");

    // we work on true binary images
    grayscale_to_binary(src_bitmap);

    // Create dst bitmap image (empty for now)
    Bitmap* dst_bitmap = createBitmap(src_bitmap->width, src_bitmap->height, src_bitmap->depth);

    // Dimensions of computing elements on the CUDA device.
    // Computing the grid dimensions depends on PAD_TOP and PAD_LEFT.
    unsigned int block_dim_x = THREADS_PER_BLOCK_X;
    unsigned int block_dim_y = THREADS_PER_BLOCK_Y;
    unsigned int grid_dim_x = (unsigned int) ceil((src_bitmap->width) / ((double) block_dim_x));
    unsigned int grid_dim_y = (unsigned int) ceil((src_bitmap->height)/ ((double) block_dim_y));
    dim3 block_dim(block_dim_x, block_dim_y);
    dim3 grid_dim(grid_dim_x, grid_dim_y);

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

    // allocate memory on device
    uint8_t* d_src_data = NULL;
    uint8_t* d_dst_data = NULL;
    unsigned int data_size = src_bitmap->width * src_bitmap->height * sizeof(uint8_t);
    cudaError d_src_malloc_success = cudaMalloc((void**) &d_src_data, data_size);
    cudaError d_dst_malloc_success = cudaMalloc((void**) &d_dst_data, data_size);
    assert((d_src_malloc_success == cudaSuccess) && "Could not allocate memory for d_src_data");
    assert((d_dst_malloc_success == cudaSuccess) && "Could not allocate memory for d_dst_data");

    // send data to device
    cudaMemcpy(d_src_data, src_bitmap->data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst_data, dst_bitmap->data, data_size, cudaMemcpyHostToDevice);

    // iterative thinning algorithm
    unsigned int iterations = 0;
    do {
        skeletonize_pass<<<grid_dim, block_dim>>>(d_src_data, d_dst_data, src_bitmap->width, src_bitmap->height, padding);

        // bring data back from device
        cudaMemcpy(src_bitmap->data, d_src_data, data_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(dst_bitmap->data, d_dst_data, data_size, cudaMemcpyDeviceToHost);

        swap_bitmaps((void**) &d_src_data, (void**) &d_dst_data);

        iterations++;
        printf(".");
        fflush(stdout);
    } while (!are_identical_bitmaps(src_bitmap, dst_bitmap));

    cudaMemcpy(dst_bitmap->data, d_dst_data, data_size, cudaMemcpyDeviceToHost);

    // Remove extra padding that was added to the images (don't care about
    // src_bitmap, so only need to unpad dst_bitmap)
    unpad_binary_bitmap(&dst_bitmap, padding);

    // save 8-bit binary-valued grayscale version of dst_bitmap to dst_fname
    binary_to_grayscale(dst_bitmap);
    int save_successful = saveBitmap(dst_fname, dst_bitmap);
    assert(save_successful && "Could not save dst_bitmap");

    // deallocate memory used for bitmaps
    cudaFree(d_src_data);
    cudaFree(d_dst_data);
    free(src_bitmap);
    free(dst_bitmap);

    return iterations;
}

// Performs 1 iteration of the thinning algorithm.
__global__ void skeletonize_pass(uint8_t* d_src, uint8_t* d_dst, unsigned int width, unsigned int height, Padding padding) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y + padding.top;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x + padding.left;

    uint8_t NZ = black_neighbors_around(d_src, row, col, width);
    uint8_t TR_P1 = wb_transitions_around(d_src, row, col, width);
    uint8_t TR_P2 = wb_transitions_around(d_src, row-1, col, width);
    uint8_t TR_P4 = wb_transitions_around(d_src, row, col-1, width);
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

int main(void) {
    for (unsigned int i = 0; i < NUMBER_OF_FILES; i++) {
        int file_name_len = strlen(src_file_names[i]);
        char* file_name = (char*) calloc(file_name_len + 1, sizeof(char));
        strncpy(file_name, src_file_names[i], file_name_len);
        char* base_file_name = basename(file_name);
        printf("%s ", base_file_name);
        fflush(stdout);
        free(file_name);

        unsigned int iterations = skeletonize(src_file_names[i], dst_file_names[i]);

        printf(" %d iterations\n", iterations);
        fflush(stdout);
    }

    printf("done\n");
    fflush(stdout);

    return 0;
}
