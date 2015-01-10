#include <assert.h>
#include <stdint.h>
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
    assert((save_successful == 1) && "Error: could not save dst_bitmap");

    // free memory used for bitmaps
    free(*src_bitmap);
    free(*dst_bitmap);

    cudaDeviceReset();
}

void gpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, dim3* grid_dim, dim3* block_dim) {
    assert((argc == 5) && "Usage: ./<gpu_binary> <input_file_name.bmp> <output_file_name.bmp> <block_size> <grid_size>");

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
    char* block_size_string = argv[3];
    char* grid_size_string = argv[4];

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
    int block_size = strtol(block_size_string, NULL, 10);
    int grid_size = strtol(grid_size_string, NULL, 10);
    assert((block_size >= 1) && "Error: block_size must be >= 1");
    assert((grid_size >= 1) && "Error: grid_size must be >= 1");
    assert((block_size <= cuda_device_properties.maxThreadsPerBlock) && "Error: block_size is larger than maxThreadsPerBlock");

    block_dim->x = block_size;
    block_dim->y = 1;
    block_dim->z = 1;
    grid_dim->x = grid_size;
    grid_dim->y = 1;
    grid_dim->z = 1;

    printf("image information\n");
    printf("=================\n");
    printf("    width = %u\n", (*src_bitmap)->width);
    printf("    height = %u\n", (*src_bitmap)->height);
    printf("    white pixels = %d %%\n", (int) (percentage_white_pixels(*src_bitmap) * 100));
    printf("    black pixels = %d %%\n", (int) (percentage_black_pixels(*src_bitmap) * 100));
    printf("\n");

    printf("cuda runtime information\n");
    printf("========================\n");
    printf("    block size = %u\n", block_size);
    printf("    grid size = %u\n", grid_size);
    printf("\n");
}

uint8_t is_power_of_2(uint8_t x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}
