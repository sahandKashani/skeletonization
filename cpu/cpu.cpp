#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpu.hpp"
#include "../common/utils.hpp"

#define PAD_TOP 2
#define PAD_LEFT 2
#define PAD_BOTTOM 1
#define PAD_RIGHT 1

#define P2(data, row, col, width) ((data)[((row)-1) * width +  (col)   ])
#define P3(data, row, col, width) ((data)[((row)-1) * width + ((col)-1)])
#define P4(data, row, col, width) ((data)[ (row)    * width + ((col)-1)])
#define P5(data, row, col, width) ((data)[((row)+1) * width + ((col)-1)])
#define P6(data, row, col, width) ((data)[((row)+1) * width +  (col)   ])
#define P7(data, row, col, width) ((data)[((row)+1) * width + ((col)+1)])
#define P8(data, row, col, width) ((data)[ (row)    * width + ((col)+1)])
#define P9(data, row, col, width) ((data)[((row)-1) * width + ((col)+1)])

// Computes the number of black neighbors around a pixel.
uint8_t black_neighbors_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width) {
    uint8_t count = 0;

    count += (P2(data, row, col, width) == BINARY_BLACK);
    count += (P3(data, row, col, width) == BINARY_BLACK);
    count += (P4(data, row, col, width) == BINARY_BLACK);
    count += (P5(data, row, col, width) == BINARY_BLACK);
    count += (P6(data, row, col, width) == BINARY_BLACK);
    count += (P7(data, row, col, width) == BINARY_BLACK);
    count += (P8(data, row, col, width) == BINARY_BLACK);
    count += (P9(data, row, col, width) == BINARY_BLACK);

    return count;
}

// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding padding) {
    unsigned int iterations = 0;

    do {
        skeletonize_pass((*src_bitmap)->data, (*dst_bitmap)->data, (*src_bitmap)->width, (*src_bitmap)->height, padding);
        swap_bitmaps((void**) src_bitmap, (void**) dst_bitmap);

        iterations++;
        printf(".");
        fflush(stdout);
    } while (!are_identical_bitmaps(*src_bitmap, *dst_bitmap));

    return iterations;
}

// Performs 1 iteration of the thinning algorithm.
void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, unsigned int height, Padding padding) {
    for (unsigned int row = padding.top; row < height - padding.bottom; row++) {
        for (unsigned int col = padding.left; col < width - padding.right; col++) {
            uint8_t NZ = black_neighbors_around(src, row, col, width);
            uint8_t TR_P1 = wb_transitions_around(src, row, col, width);
            uint8_t TR_P2 = wb_transitions_around(src, row-1, col, width);
            uint8_t TR_P4 = wb_transitions_around(src, row, col-1, width);
            uint8_t P2 = P2(src, row, col, width);
            uint8_t P4 = P4(src, row, col, width);
            uint8_t P6 = P6(src, row, col, width);
            uint8_t P8 = P8(src, row, col, width);

            uint8_t thinning_cond_1 = ((2 <= NZ) && (NZ <= 6));
            uint8_t thinning_cond_2 = (TR_P1 == 1);
            uint8_t thinning_cond_3 = (((P2 && P4 && P8) == 0) || (TR_P2 != 1));
            uint8_t thinning_cond_4 = (((P2 && P4 && P6) == 0) || (TR_P4 != 1));

            if (thinning_cond_1 && thinning_cond_2 && thinning_cond_3 && thinning_cond_4) {
                dst[row * width + col] = BINARY_WHITE;
            } else {
                dst[row * width + col] = src[row * width + col];
            }
        }
    }
}

// Computes the number of white to black transitions around a pixel.
uint8_t wb_transitions_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width) {
    uint8_t count = 0;

    count += ( (P2(data, row, col, width) == BINARY_WHITE) && (P3(data, row, col, width) == BINARY_BLACK) );
    count += ( (P3(data, row, col, width) == BINARY_WHITE) && (P4(data, row, col, width) == BINARY_BLACK) );
    count += ( (P4(data, row, col, width) == BINARY_WHITE) && (P5(data, row, col, width) == BINARY_BLACK) );
    count += ( (P5(data, row, col, width) == BINARY_WHITE) && (P6(data, row, col, width) == BINARY_BLACK) );
    count += ( (P6(data, row, col, width) == BINARY_WHITE) && (P7(data, row, col, width) == BINARY_BLACK) );
    count += ( (P7(data, row, col, width) == BINARY_WHITE) && (P8(data, row, col, width) == BINARY_BLACK) );
    count += ( (P8(data, row, col, width) == BINARY_WHITE) && (P9(data, row, col, width) == BINARY_BLACK) );
    count += ( (P9(data, row, col, width) == BINARY_WHITE) && (P2(data, row, col, width) == BINARY_BLACK) );

    return count;
}

int main(int argc, char** argv) {
    assert(argc == 3 && "Usage: cpu <input_file_name.bmp> <output_file_name.bmp>");

    char* src_fname = argv[1];
    char* dst_fname = argv[2];

    printf("src_fname = %s\n", src_fname);
    printf("dst_fname = %s\n", dst_fname);

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

    // Pad the binary images with pixels on each side. This will be useful when
    // implementing the skeletonization algorithm, because the mask we use
    // depends on P2 and P4, which also have their own window.
    Padding padding;
    padding.top = PAD_TOP;
    padding.bottom = PAD_BOTTOM;
    padding.left = PAD_LEFT;
    padding.right = PAD_RIGHT;
    pad_binary_bitmap(&src_bitmap, BINARY_WHITE, padding);
    pad_binary_bitmap(&dst_bitmap, BINARY_WHITE, padding);

    unsigned int iterations = skeletonize(&src_bitmap, &dst_bitmap, padding);
    printf(" %u iterations\n", iterations);

    // Remove extra padding that was added to the images (don't care about
    // src_bitmap, so only need to unpad dst_bitmap)
    unpad_binary_bitmap(&dst_bitmap, padding);

    // save 8-bit binary-valued grayscale version of dst_bitmap to dst_fname
    binary_to_grayscale(dst_bitmap);
    int save_successful = saveBitmap(dst_fname, dst_bitmap);
    assert(save_successful == 1 && "Error: could not save dst bitmap");

    // deallocate memory used for bitmaps
    free(src_bitmap);
    free(dst_bitmap);

    return EXIT_SUCCESS;
}
