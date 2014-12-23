#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu_only_utils.cuh"
#include "lspbmp.hpp"
#include "utils.hpp"

void gpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding) {
    char* dst_fname = argv[2];

    // Remove extra padding that was added to the images (don't care about
    // src_bitmap, so only need to unpad dst_bitmap)
    unpad_binary_bitmap(dst_bitmap, *padding);

    // save 8-bit binary-valued grayscale version of dst_bitmap to dst_fname
    binary_to_grayscale(*dst_bitmap);
    int save_successful = saveBitmap(dst_fname, *dst_bitmap);
    assert(save_successful == 1 && "Error: could not save dst_bitmap");

    // free memory used for bitmaps
    free(*src_bitmap);
    free(*dst_bitmap);

    cudaDeviceReset();
}

void gpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding, dim3* grid_dim, dim3* block_dim) {
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
    // Computing the grid dimensions depends on PAD_TOP and PAD_LEFT.
    unsigned int block_dim_x = strtol(block_dim_x_string, NULL, 10);
    unsigned int block_dim_y = strtol(block_dim_y_string, NULL, 10);
    assert((block_dim_x * block_dim_y) <= MAX_THREADS_PER_BLOCK);

    unsigned int grid_dim_x = (unsigned int) ceil(((*src_bitmap)->width) / ((double) block_dim_x));
    unsigned int grid_dim_y = (unsigned int) ceil(((*src_bitmap)->height)/ ((double) block_dim_y));
    block_dim->x = block_dim_x;
    block_dim->y = block_dim_y;
    grid_dim->x = grid_dim_x;
    grid_dim->y = grid_dim_y;

    printf("width = %u\n", (*src_bitmap)->width);
    printf("height = %u\n", (*src_bitmap)->height);

    // Pad the binary images with pixels on each side. This will be useful when
    // implementing the skeletonization algorithm, because the mask we use
    // depends on P2 and P4, which also have their own window.
    // ATTENTION : it is important to use cast to (int) since we want to test
    // for a maximum value and the subtraction can yield a negative number.
    (*padding).top = PAD_TOP;
    (*padding).bottom = max((int) ((grid_dim_y * block_dim_y) - ((*src_bitmap)->height + PAD_BOTTOM)), PAD_BOTTOM);
    (*padding).left = PAD_LEFT;
    (*padding).right = max((int) ((grid_dim_x * block_dim_x) - ((*src_bitmap)->width + PAD_RIGHT)), PAD_RIGHT);
    pad_binary_bitmap(src_bitmap, BINARY_WHITE, *padding);
    pad_binary_bitmap(dst_bitmap, BINARY_WHITE, *padding);

    printf("padded width = %u\n", (*src_bitmap)->width);
    printf("padded height = %u\n", (*src_bitmap)->height);
    printf("block dim X = %u\n", block_dim_x);
    printf("block dim Y = %u\n", block_dim_y);
    printf("grid dim X = %u\n", grid_dim_x);
    printf("grid dim Y = %u\n", grid_dim_y);
}
