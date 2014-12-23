#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "lspbmp.hpp"
#include "utils.hpp"

// Returns 1 if the 2 input bitmaps are equal, otherwise returns 0.
uint8_t are_identical_bitmaps(Bitmap* src, Bitmap* dst) {
    assert(src && "src bitmap must be non-NULL");
    assert(dst && "dst bitmap must be non-NULL");
    assert(src->width == dst->width && "src and dst must have same width");
    assert(src->height == dst->height && "src and dst must have same height");
    assert(src->depth == dst->depth && "src and dst must have same depth");

    for (unsigned int row = 0; row < src->height; row++) {
        for (unsigned int col = 0; col < src->width; col++) {
            if (dst->data[row * src->width + col] != src->data[row * src->width + col]) {
                return 0;
            }
        }
    }

    return 1;
}

// Converts an 8-bit binary image (black = 1, white = 0) to an 8-bit
// binary-valued grayscale image (black = 0, white = 255).
void binary_to_grayscale(Bitmap* image) {
    assert(image && "Bitmap must be non-NULL");
    assert(is_binary_image(image) && "Must supply a binary image as input: only black (1) and white (0) are allowed");

    for (unsigned int row = 0; row < image->height; row++) {
        for (unsigned int col = 0; col < image->width; col++) {
            uint8_t value = image->data[row * image->width + col];
            image->data[row * image->width + col] = (value == BINARY_BLACK) ? GRAYSCALE_BLACK : GRAYSCALE_WHITE;
        }
    }
}

// Copies the src bitmap to the dst bitmap.
void copy_bitmap(Bitmap* src, Bitmap* dst) {
    assert(src && "src bitmap must be non-NULL");
    assert(dst && "dst bitmap must be non-NULL");
    assert(src->width == dst->width && "src and dst must have same width");
    assert(src->height == dst->height && "src and dst must have same height");
    assert(src->depth == dst->depth && "src and dst must have same depth");

    for (unsigned int row = 0; row < src->height; row++) {
        for (unsigned int col = 0; col < src->width; col++) {
            dst->data[row * src->width + col] = src->data[row * src->width + col];
        }
    }
}

// Converts an 8-bit binary-valued grayscale image (black = 0, white = 255) to a
// 1-bit binary image (black = 1, white = 0). The image is still stored on 8
// bits, but the values reflect those of a 1-bit binary image.
void grayscale_to_binary(Bitmap* image) {
    assert(image && "Bitmap must be non-NULL");
    assert(is_binary_valued_grayscale_image(image) && "Must supply a binary-valued grayscale image: only black (0) and white (255) are allowed");

    for (unsigned int row = 0; row < image->height; row++) {
        for (unsigned int col = 0; col < image->width; col++) {
            uint8_t value = image->data[row * image->width + col];
            image->data[row * image->width + col] = (value == GRAYSCALE_BLACK) ? BINARY_BLACK : BINARY_WHITE;
        }
    }
}

// Checks if the input image is a binary image. Returns 1 if the input is an
// 8-bit binary image (black = 1, white = 0). Returns 0 otherwise.
uint8_t is_binary_image(Bitmap* image) {
    assert(image && "Bitmap must be non-NULL");

    // check 8-bit depth (even for a binary value)
    if (image->depth != 8) {
        return 0;
    }

    for (unsigned int row = 0; row < image->height; row++) {
        for (unsigned int col = 0; col < image->width; col++) {
            uint8_t value = image->data[row * image->width + col];
            if (!(value == BINARY_BLACK || value == BINARY_WHITE)) {
                return 0;
            }
        }
    }

    return 1;
}

// Checks if the input image is a binary-valued grayscale image. Returns 1 if
// the input is an 8-bit grayscale bitmap image. Additionally, the only possible
// values should be black (0) and white (255). Returns 0 otherwise.
uint8_t is_binary_valued_grayscale_image(Bitmap* image) {
    assert(image && "Bitmap must be non-NULL");

    // check 8-bit depth
    if (image->depth != 8) {
        return 0;
    }

    for (unsigned int row = 0; row < image->height; row++) {
        for (unsigned int col = 0; col < image->width; col++) {
            uint8_t value = image->data[row * image->width + col];
            if (!(value == GRAYSCALE_BLACK || value == GRAYSCALE_WHITE)) {
                return 0;
            }
        }
    }

    return 1;
}

uint8_t is_outside_image(int row, int col, unsigned int width, unsigned int height) {
    return (row <= 0) || (row >= (height - 1)) || (col <= 0) || (col >= (width - 1));
}

// Prints information about a bitmap image.
void print_bitmap_info(const char* fname) {
    assert(fname && "Invalid file name");

    Bitmap* bitmap = loadBitmap(fname);
    assert(bitmap && "Bitmap has to be non-NULL");

    printf("%s\n", fname);
    printf("    width  = %u\n", bitmap->width);
    printf("    height = %u\n", bitmap->height);
    printf("    depth  = %u\n", bitmap->depth);

    free(bitmap);
}

// Swaps 2 pointers.
void swap_bitmaps(void** src, void** dst) {
    void* tmp = *dst;
    *dst = *src;
    *src = tmp;
}
