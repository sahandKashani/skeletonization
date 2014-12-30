#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "cpu_only_utils.hpp"
#include "lspbmp.hpp"
#include "utils.hpp"

void cpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap) {
    char* dst_fname = argv[2];

    // save 8-bit binary-valued grayscale version of dst_bitmap to dst_fname
    binary_to_grayscale(*dst_bitmap);
    int save_successful = saveBitmap(dst_fname, *dst_bitmap);
    assert((save_successful == 1) && "Error: could not save dst_bitmap");

    // free memory used for bitmaps
    free(*src_bitmap);
    free(*dst_bitmap);
}

void cpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap) {
    assert((argc == 3) && "Usage: ./<cpu_binary> <input_file_name.bmp> <output_file_name.bmp>");

    char* src_fname = argv[1];
    char* dst_fname = argv[2];

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

    printf("image information\n");
    printf("=================\n");
    printf("    width = %u\n", (*src_bitmap)->width);
    printf("    height = %u\n", (*src_bitmap)->height);
    printf("    white pixels = %d%%\n", (int) (percentage_white_pixels(*src_bitmap) * 100));
    printf("    black pixels = %d%%\n", (int) (percentage_black_pixels(*src_bitmap) * 100));
    printf("\n");
}
