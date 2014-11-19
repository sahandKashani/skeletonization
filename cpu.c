#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "cpu.h"

// Prints information about a bitmap image.
void print_bitmap_info(const char* fname) {
    assert(fname && "Invalid file name");

    Bitmap* bitmap = loadBitmap(fname);
    assert(bitmap && "Bitmap has to be non-NULL");

    printf("%s\n", fname);
    printf("    width  = %d\n", bitmap->width);
    printf("    height = %d\n", bitmap->height);
    printf("    depth  = %d\n", bitmap->depth);

    free(bitmap);
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

// Unpads the image given as input by removing the amount of padding provided as
// input.
void unpad_bitmap(Bitmap** image, Padding padding) {
    assert(*image && "Bitmap must be non-NULL");

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

// Computes the number of white to black transitions around a pixel.
uint8_t wb_transitions_around(Bitmap* bitmap, unsigned int row, unsigned int col) {
    uint8_t count = 0;

    count += ( (P2(bitmap, row, col) == BINARY_WHITE) && (P3(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P3(bitmap, row, col) == BINARY_WHITE) && (P4(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P4(bitmap, row, col) == BINARY_WHITE) && (P5(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P5(bitmap, row, col) == BINARY_WHITE) && (P6(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P6(bitmap, row, col) == BINARY_WHITE) && (P7(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P7(bitmap, row, col) == BINARY_WHITE) && (P8(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P8(bitmap, row, col) == BINARY_WHITE) && (P9(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P9(bitmap, row, col) == BINARY_WHITE) && (P2(bitmap, row, col) == BINARY_BLACK) );

    return count;
}

// Computes the number of black neighbors around a pixel.
uint8_t black_neighbors_around(Bitmap* bitmap, unsigned int row, unsigned int col) {
    uint8_t count = 0;

    count += (P2(bitmap, row, col) == BINARY_BLACK);
    count += (P3(bitmap, row, col) == BINARY_BLACK);
    count += (P4(bitmap, row, col) == BINARY_BLACK);
    count += (P5(bitmap, row, col) == BINARY_BLACK);
    count += (P6(bitmap, row, col) == BINARY_BLACK);
    count += (P7(bitmap, row, col) == BINARY_BLACK);
    count += (P8(bitmap, row, col) == BINARY_BLACK);
    count += (P9(bitmap, row, col) == BINARY_BLACK);

    return count;
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

// Performs 1 iteration of the thinning algorithm.
void skeletonize_pass(Bitmap* src, Bitmap* dst, Padding padding) {
    assert(src && "src bitmap must be non-NULL");
    assert(dst && "dst bitmap must be non-NULL");
    assert(src->width == dst->width && "src and dst must have same width");
    assert(src->height == dst->height && "src and dst must have same height");
    assert(src->depth == dst->depth && "src and dst must have same depth");

    for (unsigned int row = padding.top; row < src->height - padding.bottom; row++) {
        for (unsigned int col = padding.left; col < src->width - padding.right; col++) {
            uint8_t NZ = black_neighbors_around(src, row, col);
            uint8_t TR_P1 = wb_transitions_around(src, row, col);
            uint8_t TR_P2 = wb_transitions_around(src, row-1, col);
            uint8_t TR_P4 = wb_transitions_around(src, row, col-1);
            uint8_t P2 = P2(src, row, col);
            uint8_t P4 = P4(src, row, col);
            uint8_t P6 = P6(src, row, col);
            uint8_t P8 = P8(src, row, col);

            uint8_t thinning_cond_1 = ((2 <= NZ) && (NZ <= 6));
            uint8_t thinning_cond_2 = (TR_P1 == 1);
            uint8_t thinning_cond_3 = (((P2 && P4 && P8) == 0) || (TR_P2 != 1));
            uint8_t thinning_cond_4 = (((P2 && P4 && P6) == 0) || (TR_P4 != 1));

            if (thinning_cond_1 && thinning_cond_2 && thinning_cond_3 && thinning_cond_4) {
                dst->data[row * src->width + col] = BINARY_WHITE;
            } else {
                dst->data[row * src->width + col] = src->data[row * src->width + col];
            }
        }
    }
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

    // Pad the binary images with pixels on each side. This will be useful when
    // implementing the skeletonization algorithm, because the mask we use
    // depends on P2 and P4, which also have their own window.
    Padding padding_amounts;
    padding_amounts.top = PAD_TOP;
    padding_amounts.bottom = PAD_BOTTOM;
    padding_amounts.left = PAD_LEFT;
    padding_amounts.right = PAD_RIGHT;

    pad_binary_bitmap(&src_bitmap, BINARY_WHITE, padding_amounts);
    pad_binary_bitmap(&dst_bitmap, BINARY_WHITE, padding_amounts);

    // iterative thinning algorithm
    unsigned int iterations = 0;
    do {
        skeletonize_pass(src_bitmap, dst_bitmap, padding_amounts);
        swap_bitmaps(&src_bitmap, &dst_bitmap);

        iterations++;
        printf("iteration %d\n", iterations);
    } while (!are_identical_bitmaps(src_bitmap, dst_bitmap));

    // Remove extra padding that was added to the images (don't care about
    // src_bitmap, so only need to unpad dst_bitmap)
    unpad_bitmap(&dst_bitmap, padding_amounts);

    // save 8-bit binary-valued grayscale version of dst_bitmap to dst_fname
    binary_to_grayscale(dst_bitmap);
    int save_successful = saveBitmap(dst_fname, dst_bitmap);
    assert(save_successful && "Could not save dst_bitmap");

    // deallocate memory used for bitmaps
    free(src_bitmap);
    free(dst_bitmap);

    return iterations;
}

// Swaps 2 bitmaps by exchanging their pointers.
void swap_bitmaps(Bitmap** src, Bitmap** dst) {
    Bitmap* tmp = *dst;
    *dst = *src;
    *src = tmp;
}

int main(void) {
    unsigned int image_index = 0;
    skeletonize(src_file_names[image_index], dst_file_names[image_index]);

//    for (unsigned int i = 0; i < NUMBER_OF_FILES; i++) {
//        unsigned int iterations = skeletonize(src_file_names[i], dst_file_names[i]);
//        printf("%s\n", src_file_names[i]);
//        printf("    %d iterations\n", iterations);
//    }

    printf("done\n");
    return 0;
}
