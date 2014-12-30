#include <assert.h>
#include <stdint.h>
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
    assert((save_successful == 1) && "Error: could not save dst_bitmap");

    // free memory used for bitmaps
    free(*src_bitmap);
    free(*dst_bitmap);

    cudaDeviceReset();
}

void gpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding, dim3* grid_dim, dim3* block_dim) {
    assert((argc == 5) && "Usage: ./<gpu_binary> <input_file_name.bmp> <output_file_name.bmp> <block_dim_x> <block_dim_y>");

    // check for cuda-capable device
    int cuda_device_count;
    gpuErrchk(cudaGetDeviceCount(&cuda_device_count));
    assert((cuda_device_count > 0) && "Error: no CUDA-capable device detected");

    // select a cuda-capable device
    int cuda_device_id;
    cudaDeviceProp cuda_device_properties;
    memset(&cuda_device_properties, 0, sizeof(cudaDeviceProp));
    // Would like any cuda-capable device with compute capability 2.0 (because
    // the compiler generates code with abnormally high register usage for 1.0
    // devices)
    cuda_device_properties.major = 2;
    cuda_device_properties.minor = 0;
    gpuErrchk(cudaChooseDevice(&cuda_device_id, &cuda_device_properties));
    gpuErrchk(cudaSetDevice(cuda_device_id));

    // print some info about the chosen GPU
    gpuErrchk(cudaGetDeviceProperties(&cuda_device_properties, cuda_device_id));
    printf("cuda device information\n");
    printf("=======================\n");
    printf("    device name = %s\n", cuda_device_properties.name);
    printf("    Compute capability = %d.%d\n", cuda_device_properties.major, cuda_device_properties.minor);
    printf("    multiprocessor count = %d\n", cuda_device_properties.multiProcessorCount);
    printf("    total global memory = %lu\n", cuda_device_properties.totalGlobalMem);
    printf("    total constant memory = %lu\n", cuda_device_properties.totalConstMem);
    printf("    shared memory per block = %lu\n", cuda_device_properties.sharedMemPerBlock);
    printf("    registers per block = %d\n", cuda_device_properties.regsPerBlock);
    printf("    max threads per block = %d\n", cuda_device_properties.maxThreadsPerBlock);
    printf("    max thread dimensions = (%d, %d, %d)\n", cuda_device_properties.maxThreadsDim[0], cuda_device_properties.maxThreadsDim[1], cuda_device_properties.maxThreadsDim[2]);
    printf("    max grid dimensions = (%d, %d, %d)\n", cuda_device_properties.maxGridSize[0], cuda_device_properties.maxGridSize[1], cuda_device_properties.maxGridSize[2]);
    printf("    warp size = %d\n", cuda_device_properties.warpSize);
    printf("    kernel execution timeout = ");
    if (cuda_device_properties.kernelExecTimeoutEnabled) {
        printf("true\n");
    } else {
        printf("false\n");
    }
    printf("\n");

    char* src_fname = argv[1];
    char* dst_fname = argv[2];
    char* block_dim_x_string = argv[3];
    char* block_dim_y_string = argv[4];

    printf("src_fname = %s\n", src_fname);
    printf("dst_fname = %s\n", dst_fname);
    printf("\n");

    // load src image
    *src_bitmap = loadBitmap(src_fname);
    assert((*src_bitmap != NULL) && "Error: could not load src_bitmap");

    // validate src image is 8-bit binary-valued grayscale image
    assert(is_binary_valued_grayscale_image(*src_bitmap) && "Error: Only 8-bit binary-valued grayscale images are supported. Values must be black (0) or white (255) only");

    // we work on true binary images
    grayscale_to_binary(*src_bitmap);

    // Create dst bitmap image (empty for now)
    *dst_bitmap = createBitmap((*src_bitmap)->width, (*src_bitmap)->height, (*src_bitmap)->depth);
    assert((*dst_bitmap != NULL) && "Error: could not allocate memory for dst_bitmap");

    // Dimensions of computing elements on the CUDA device.
    int block_dim_x = strtol(block_dim_x_string, NULL, 10);
    int block_dim_y = strtol(block_dim_y_string, NULL, 10);
    assert(((block_dim_x * block_dim_y) <= cuda_device_properties.maxThreadsPerBlock) && "Error: Using more threads than permitted by maxThreadsPerBlock");
    // TODO : enable this
    // assert((((block_dim_x * block_dim_y) % cuda_device_properties.warpSize) == 0) && "Error: Must use thread count which is a multiple of warpSize");

    int grid_dim_x = (int) ceil(((*src_bitmap)->width) / ((double) block_dim_x));
    int grid_dim_y = (int) ceil(((*src_bitmap)->height)/ ((double) block_dim_y));
    block_dim->x = block_dim_x;
    block_dim->y = block_dim_y;
    block_dim->z = 1;
    grid_dim->x = grid_dim_x;
    grid_dim->y = grid_dim_y;
    grid_dim->z = 1;

    printf("image information\n");
    printf("=================\n");
    printf("    width = %u\n", (*src_bitmap)->width);
    printf("    height = %u\n", (*src_bitmap)->height);
    printf("    white pixels = %d%%\n", (int) (percentage_white_pixels(*src_bitmap) * 100));
    printf("    black pixels = %d%%\n", (int) (percentage_black_pixels(*src_bitmap) * 100));
    printf("\n");

    printf("cuda runtime information\n");
    printf("========================\n");
    printf("    block dim X = %u\n", block_dim_x);
    printf("    block dim Y = %u\n", block_dim_y);
    printf("    grid dim X = %u\n", grid_dim_x);
    printf("    grid dim Y = %u\n", grid_dim_y);
    printf("\n");

    // Pad the binary images with pixels on the right and bottom. This will be
    // useful when implementing the skeletonization algorithm, as we can make
    // sure that all threads have some data to work on (even if it is bogus)
    (*padding).bottom = (grid_dim_y * block_dim_y) - ((*src_bitmap)->height);
    (*padding).right = (grid_dim_x * block_dim_x) - ((*src_bitmap)->width);
    pad_binary_bitmap(src_bitmap, BINARY_WHITE, *padding);
    pad_binary_bitmap(dst_bitmap, BINARY_WHITE, *padding);

    printf("image information after padding\n");
    printf("===============================\n");
    printf("    width = %u\n", (*src_bitmap)->width);
    printf("    height = %u\n", (*src_bitmap)->height);
    printf("    white pixels = %d%%\n", (int) (percentage_white_pixels(*src_bitmap) * 100));
    printf("    black pixels = %d%%\n", (int) (percentage_black_pixels(*src_bitmap) * 100));
    printf("\n");
}

// Pads the binary image given as input with the padding values provided as
// input. The padding value must be a binary white (0) or black (1).
void pad_binary_bitmap(Bitmap** image, uint8_t binary_padding_value, Padding padding) {
    assert((*image != NULL) && "Error: Bitmap must be non-NULL");
    assert(is_binary_image(*image) && "Error: Must supply a binary image as input: only black (1) and white (0) are allowed");
    assert((binary_padding_value == BINARY_BLACK || binary_padding_value == BINARY_WHITE) && "Error: Must provide a binary value for padding");

    // allocate buffer for image data with extra rows and extra columns
    Bitmap *new_image = createBitmap((*image)->width + padding.right, (*image)->height + padding.bottom, (*image)->depth);

    // copy original data into the center of the new buffer
    for (int row = 0; row < new_image->height; row++) {
        for (int col = 0; col < new_image->width; col++) {

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
    assert((*image != NULL) && "Error: Bitmap must be non-NULL");
    assert(is_binary_image(*image) && "Error: Must supply a binary image as input: only black (1) and white (0) are allowed");

    // allocate buffer for image data with less rows and less columns
    Bitmap *new_image = createBitmap((*image)->width - padding.right, (*image)->height - padding.bottom, (*image)->depth);

    // copy data from larger image into the middle of the new buffer
    for (int row = 0; row < new_image->height; row++) {
        for (int col = 0; col < new_image->width; col++) {
            new_image->data[row * new_image->width + col] = (*image)->data[row * ((*image)->width) + col];
        }
    }

    free(*image);
    *image = new_image;
}
