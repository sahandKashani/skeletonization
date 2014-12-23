#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu_only_utils.cuh"
#include "lspbmp.hpp"
#include "utils.hpp"

void gpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap) {
    char* dst_fname = argv[2];

    // save 8-bit binary-valued grayscale version of dst_bitmap to dst_fname
    binary_to_grayscale(*dst_bitmap);
    int save_successful = saveBitmap(dst_fname, *dst_bitmap);
    assert(save_successful == 1 && "Error: could not save dst_bitmap");

    // free memory used for bitmaps
    free(*src_bitmap);
    free(*dst_bitmap);

    cudaDeviceReset();
}

void gpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3* grid_dim, dim3* block_dim) {
    assert(argc == 5 && "Usage: ./<gpu_binary> <input_file_name.bmp> <output_file_name.bmp> <block_dim_x> <block_dim_y>");

    char* src_fname = argv[1];
    char* dst_fname = argv[2];
    char* block_dim_x_string = argv[3];
    char* block_dim_y_string = argv[4];

    printf("src_fname = %s\n", src_fname);
    printf("dst_fname = %s\n", dst_fname);

    // load src image
    *src_bitmap = loadBitmap(src_fname);
    assert(*src_bitmap != NULL && "Error: could not load src_bitmap");

    // validate src image is 8-bit binary-valued grayscale image
    assert(is_binary_valued_grayscale_image(*src_bitmap) && "Error: Only 8-bit binary-valued grayscale images are supported. Values must be black (0) or white (255) only");

    // we work on true binary images
    grayscale_to_binary(*src_bitmap);

    // Create dst bitmap image (empty for now)
    *dst_bitmap = createBitmap((*src_bitmap)->width, (*src_bitmap)->height, (*src_bitmap)->depth);
    assert(*dst_bitmap != NULL && "Error: could not allocate memory for dst_bitmap");

    // Dimensions of computing elements on the CUDA device.
    int block_dim_x = strtol(block_dim_x_string, NULL, 10);
    int block_dim_y = strtol(block_dim_y_string, NULL, 10);
    assert((block_dim_x * block_dim_y) <= MAX_THREADS_PER_BLOCK);

    int grid_dim_x = (int) ceil(((*src_bitmap)->width) / ((double) block_dim_x));
    int grid_dim_y = (int) ceil(((*src_bitmap)->height)/ ((double) block_dim_y));
    block_dim->x = block_dim_x;
    block_dim->y = block_dim_y;
    block_dim->z = 1;
    grid_dim->x = grid_dim_x;
    grid_dim->y = grid_dim_y;
    grid_dim->z = 1;

    printf("width = %u\n", (*src_bitmap)->width);
    printf("height = %u\n", (*src_bitmap)->height);
    printf("block dim X = %u\n", block_dim_x);
    printf("block dim Y = %u\n", block_dim_y);
    printf("grid dim X = %u\n", grid_dim_x);
    printf("grid dim Y = %u\n", grid_dim_y);
}
