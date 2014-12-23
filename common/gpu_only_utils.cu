#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "gpu_only_utils.cuh"
#include "lspbmp.hpp"
#include "utils.hpp"

void gpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding) {
    char* dst_fname = argv[2];

    // // Remove extra padding that was added to the images (don't care about
    // // src_bitmap, so only need to unpad dst_bitmap)
    // unpad_binary_bitmap(dst_bitmap, *padding);

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
    block_dim->z = 1;
    grid_dim->x = grid_dim_x;
    grid_dim->y = grid_dim_y;
    grid_dim->z = 1;

    printf("width = %u\n", (*src_bitmap)->width);
    printf("height = %u\n", (*src_bitmap)->height);

    // // Pad the binary images with pixels on the right and bottom. This will be
    // // useful when implementing the skeletonization algorithm, as we can make
    // // sure that all threads have some data to work on (even if it is bogus)
    // // ATTENTION : it is important to use cast to (int) since we want to test
    // // for a maximum value and the subtraction can yield a negative number.
    // (*padding).bottom = (grid_dim_y * block_dim_y) - ((*src_bitmap)->height);
    // (*padding).right = (grid_dim_x * block_dim_x) - ((*src_bitmap)->width);
    // pad_binary_bitmap(src_bitmap, BINARY_WHITE, *padding);
    // pad_binary_bitmap(dst_bitmap, BINARY_WHITE, *padding);

    printf("padded width = %u\n", (*src_bitmap)->width);
    printf("padded height = %u\n", (*src_bitmap)->height);
    printf("block dim X = %u\n", block_dim_x);
    printf("block dim Y = %u\n", block_dim_y);
    printf("grid dim X = %u\n", grid_dim_x);
    printf("grid dim Y = %u\n", grid_dim_y);
}

// Pads the binary image given as input with the padding values provided as
// input. The padding value must be a binary white (0) or black (1).
void pad_binary_bitmap(Bitmap** image, uint8_t binary_padding_value, Padding padding) {
    assert(*image && "Bitmap must be non-NULL");
    assert(is_binary_image(*image) && "Must supply a binary image as input: only black (1) and white (0) are allowed");
    assert((binary_padding_value == BINARY_BLACK || binary_padding_value == BINARY_WHITE) && "Must provide a binary value for padding");

    // allocate buffer for image data with extra rows and extra columns
    Bitmap *new_image = createBitmap((*image)->width + padding.right, (*image)->height + padding.bottom, (*image)->depth);

    // copy original data into the center of the new buffer
    for (unsigned int row = 0; row < new_image->height; row++) {
        for (unsigned int col = 0; col < new_image->width; col++) {

            uint8_t is_bottom_row_padding_zone = ( ((new_image->height - padding.bottom) <= row) && (row <= (new_image->height-1)) );
            uint8_t is_right_col_padding_zone = ( ((new_image->width - padding.right) <= col) && (col <= (new_image->width-1)) );

            if (is_bottom_row_padding_zone || is_right_col_padding_zone) {
                // set the border pixels around the center image to binary_padding_value
                new_image->data[row * (new_image->width) + col] = binary_padding_value;
            } else {
                // set the pixels in the center to the original image
                new_image->data[row * (new_image->width) + col] = (*image)->data[row * ((*image)->width) + col];
            }
        }
    }

    free(*image);
    *image = new_image;
}

// Unpads the image given as input by removing the amount of padding provided as
// input.
void unpad_binary_bitmap(Bitmap** image, Padding padding) {
    assert(*image && "Bitmap must be non-NULL");
    assert(is_binary_image(*image) && "Must supply a binary image as input: only black (1) and white (0) are allowed");

    // allocate buffer for image data with less rows and less columns
    Bitmap *new_image = createBitmap((*image)->width - padding.right, (*image)->height - padding.bottom, (*image)->depth);

    // copy data from larger image into the middle of the new buffer
    for (unsigned int row = 0; row < new_image->height; row++) {
        for (unsigned int col = 0; col < new_image->width; col++) {
            new_image->data[row * new_image->width + col] = (*image)->data[row * ((*image)->width) + col];
        }
    }

    free(*image);
    *image = new_image;
}
