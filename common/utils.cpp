#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "lspbmp.hpp"
#include "utils.hpp"

#define PAD_TOP 2
#define PAD_LEFT 2
#define PAD_BOTTOM 1
#define PAD_RIGHT 1

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

void cpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding) {
    assert(argc == 3 && "Usage: ./<cpu_binary> <input_file_name.bmp> <output_file_name.bmp>");

    char* src_fname = argv[1];
    char* dst_fname = argv[2];

    printf("src_fname = %s\n", src_fname);
    printf("dst_fname = %s\n", dst_fname);

    // load src image
    *src_bitmap = loadBitmap(src_fname);
    assert(*src_bitmap != NULL && "Error: could not load src_bitmap");

    printf("width = %u\n", (*src_bitmap)->width);
    printf("height = %u\n", (*src_bitmap)->height);

    // validate src image is 8-bit binary-valued grayscale image
    assert(is_binary_valued_grayscale_image(*src_bitmap) && "Error: Only 8-bit binary-valued grayscale images are supported. Values must be black (0) or white (255) only");

    // we work on true binary images
    grayscale_to_binary(*src_bitmap);

    // Create dst bitmap image (empty for now)
    *dst_bitmap = createBitmap((*src_bitmap)->width, (*src_bitmap)->height, (*src_bitmap)->depth);
    assert(*dst_bitmap != NULL && "Error: could not allocate memory for dst_bitmap");

    // Pad the binary images with pixels on each side. This will be useful when
    // implementing the skeletonization algorithm, because the mask we use
    // depends on P2 and P4, which also have their own window.
    (*padding).top = PAD_TOP;
    (*padding).bottom = PAD_BOTTOM;
    (*padding).left = PAD_LEFT;
    (*padding).right = PAD_RIGHT;
    pad_binary_bitmap(src_bitmap, BINARY_WHITE, *padding);
    pad_binary_bitmap(dst_bitmap, BINARY_WHITE, *padding);
}

void cpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding* padding) {
    char* dst_fname = argv[2];

    // Remove extra padding that was added to the images (don't care about
    // src_bitmap, so only need to unpad dst_bitmap)
    unpad_binary_bitmap(dst_bitmap, *padding);

    // save 8-bit binary-valued grayscale version of dst_bitmap to dst_fname
    binary_to_grayscale(*dst_bitmap);
    int save_successful = saveBitmap(dst_fname, *dst_bitmap);
    assert(save_successful == 1 && "Error: could not save dst bitmap");

    // free memory used for bitmaps
    free(*src_bitmap);
    free(*dst_bitmap);
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

// Pads the binary image given as input with the padding values provided as
// input. The padding value must be a binary white (0) or black (1).
void pad_binary_bitmap(Bitmap** image, uint8_t binary_padding_value, Padding padding) {
    assert(*image && "Bitmap must be non-NULL");
    assert(is_binary_image(*image) && "Must supply a binary image as input: only black (1) and white (0) are allowed");
    assert((binary_padding_value == BINARY_BLACK || binary_padding_value == BINARY_WHITE) && "Must provide a binary value for padding");

    // allocate buffer for image data with extra rows and extra columns
    Bitmap *new_image = createBitmap((*image)->width + (padding.left + padding.right), (*image)->height + (padding.top + padding.bottom), (*image)->depth);

    // copy original data into the center of the new buffer
    for (unsigned int row = 0; row < new_image->height; row++) {
        for (unsigned int col = 0; col < new_image->width; col++) {

            uint8_t is_top_row_padding_zone = ( (0 <= row) && (row <= padding.top) );
            uint8_t is_bottom_row_padding_zone = ( ((new_image->height - padding.bottom) <= row) && (row <= (new_image->height-1)) );
            uint8_t is_left_col_padding_zone = ( (0 <= col) && (col <= padding.left) );
            uint8_t is_right_col_padding_zone = ( ((new_image->width - padding.right) <= col) && (col <= (new_image->width-1)) );

            if (is_top_row_padding_zone || is_bottom_row_padding_zone ||
                is_left_col_padding_zone || is_right_col_padding_zone) {
                // set the border pixels around the center image to binary_padding_value
                new_image->data[row * (new_image->width) + col] = binary_padding_value;
            } else {
                // set the pixels in the center to the original image
                new_image->data[row * (new_image->width) + col] = (*image)->data[(row-padding.top) * ((*image)->width) + (col-padding.left)];
            }
        }
    }

    free(*image);
    *image = new_image;
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

// Unpads the image given as input by removing the amount of padding provided as
// input.
void unpad_binary_bitmap(Bitmap** image, Padding padding) {
    assert(*image && "Bitmap must be non-NULL");
    assert(is_binary_image(*image) && "Must supply a binary image as input: only black (1) and white (0) are allowed");

    // allocate buffer for image data with less rows and less columns
    Bitmap *new_image = createBitmap((*image)->width - (padding.left + padding.right), (*image)->height - (padding.top + padding.bottom), (*image)->depth);

    // copy data from larger image into the middle of the new buffer
    for (unsigned int row = 0; row < new_image->height; row++) {
        for (unsigned int col = 0; col < new_image->width; col++) {
            new_image->data[row * new_image->width + col] = (*image)->data[(row+padding.top) * ((*image)->width) + (col+padding.left)];
        }
    }

    free(*image);
    *image = new_image;
}
