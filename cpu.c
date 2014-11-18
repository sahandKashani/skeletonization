#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "cpu.h"

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

    uint8_t is_grayscale = 1;

    // check 8-bit depth
    is_grayscale &= (image->depth == 8);

    for (unsigned int row = 0; row < image->height; row++) {
        for (unsigned int col = 0; col < image->width; col++) {
            uint8_t value = image->data[row * image->width + col];
            is_grayscale &= (value == GRAYSCALE_BLACK || value == GRAYSCALE_WHITE);
        }
    }

    return is_grayscale;
}

// Checks if the input image is a binary image. Returns 1 if the input is an
// 8-bit binary image (black = 1, white = 0). Returns 0 otherwise.
uint8_t is_binary_image(Bitmap* image) {
    assert(image && "Bitmap must be non-NULL");

    uint8_t is_binary = 1;

    // check 8-bit depth (even for a binary value)
    is_binary &= (image->depth == 8);

    for (unsigned int row = 0; row < image->height; row++) {
        for (unsigned int col = 0; col < image->width; col++) {
            uint8_t value = image->data[row * image->width + col];
            is_binary &= (value == BINARY_BLACK || value == BINARY_WHITE);
        }
    }

    return is_binary;
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

// Pads the binary image given as input with 2*padding_amount extra rows and
// 2*padding_amount extra columns, padding_amount on each side. The padding
// value must be a binary white (0) or black (1).
void pad_binary_bitmap(Bitmap** image, uint8_t binary_padding_value, uint8_t padding_amount) {
    assert(*image && "Bitmap must be non-NULL");
    assert(is_binary_image(*image) && "Must supply a binary image as input: only black (1) and white (0) are allowed");
    assert((binary_padding_value == BINARY_BLACK || binary_padding_value == BINARY_WHITE) && "Must provide a binary value for padding");

    // allocate buffer for image data with 2 extra rows and 2 extra columns
    Bitmap *new_image = createBitmap((*image)->width + (2*padding_amount), (*image)->height + (2*padding_amount), (*image)->depth);

    // copy original data into the middle of the new buffer
    for (unsigned int row = 0; row < new_image->height; row++) {
        for (unsigned int col = 0; col < new_image->width; col++) {

            uint8_t is_lower_row_padding_zone = ( (0 <= row) && (row <= (padding_amount-1)) );
            uint8_t is_upper_row_padding_zone = ( ((new_image->height - padding_amount) <= row) && (row <= (new_image->height-1)) );
            uint8_t is_lower_col_padding_zone = ( (0 <= col) && (col <= (padding_amount-1)) );
            uint8_t is_upper_col_padding_zone = ( ((new_image->width - padding_amount) <= col) && (col <= (new_image->width-1)) );

            if (is_lower_row_padding_zone || is_upper_row_padding_zone ||
                is_lower_col_padding_zone || is_upper_col_padding_zone) {
                // set the border pixels around the center image to binary white
                new_image->data[row * (new_image->width) + col] = binary_padding_value;
            } else {
                // set the pixels in the center to the original image
                new_image->data[row * (new_image->width) + col] = (*image)->data[(row-padding_amount) * ((*image)->width) + (col-padding_amount)];
            }
        }
    }

    free(*image);
    *image = new_image;
}

// Unpads the image given as input by removing 2*padding_amount extra rows and
// 2*padding_amount extra columns, removing padding_amount on each side.
void unpad_bitmap(Bitmap** image, uint8_t padding_amount) {
    assert(*image && "Bitmap must be non-NULL");

    // allocate buffer for image data with 2 extra rows and 2 extra columns
    Bitmap *new_image = createBitmap((*image)->width - (2*padding_amount), (*image)->height - (2*padding_amount), (*image)->depth);

    // copy original data into the middle of the new buffer
    for (unsigned int row = 0; row < new_image->height; row++) {
        for (unsigned int col = 0; col < new_image->width; col++) {
            // set the pixels in the center to the orignal image
            new_image->data[row * new_image->width + col] = (*image)->data[(row+padding_amount) * ((*image)->width) + (col+padding_amount)];
        }
    }

    free(*image);
    *image = new_image;
}

uint8_t wb_transitions_around(Bitmap* bitmap, unsigned int row, unsigned int col) {
    uint8_t count = 0;

    count += ( (P2(bitmap, row, col) == BINARY_WHITE) & (P3(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P3(bitmap, row, col) == BINARY_WHITE) & (P4(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P4(bitmap, row, col) == BINARY_WHITE) & (P5(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P5(bitmap, row, col) == BINARY_WHITE) & (P6(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P6(bitmap, row, col) == BINARY_WHITE) & (P7(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P7(bitmap, row, col) == BINARY_WHITE) & (P8(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P8(bitmap, row, col) == BINARY_WHITE) & (P9(bitmap, row, col) == BINARY_BLACK) );
    count += ( (P9(bitmap, row, col) == BINARY_WHITE) & (P2(bitmap, row, col) == BINARY_BLACK) );

    return count;
}

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

uint8_t are_identical_bitmaps(Bitmap* src, Bitmap* dst) {
    assert(src && "src bitmap must be non-NULL");
    assert(dst && "dst bitmap must be non-NULL");
    assert(src->width == dst->width && "src and dst must have same width");
    assert(src->height == dst->height && "src and dst must have same height");
    assert(src->depth == dst->depth && "src and dst must have same depth");

    uint8_t identical = 1;

    for (unsigned int row = 0; row < src->height; row++) {
        for (unsigned int col = 0; col < src->width; col++) {
            if (dst->data[row * src->width + col] != src->data[row * src->width + col]) {
                identical = 0;
            }
        }
    }

    return identical;
}

void skeletonize_pass(Bitmap* src, Bitmap* dst) {
    assert(src && "src bitmap must be non-NULL");
    assert(dst && "dst bitmap must be non-NULL");
    assert(src->width == dst->width && "src and dst must have same width");
    assert(src->height == dst->height && "src and dst must have same height");
    assert(src->depth == dst->depth && "src and dst must have same depth");

    for (unsigned int row = PAD_SIZE; row < src->height - PAD_SIZE; row++) {
        for (unsigned int col = PAD_SIZE; col < src->width - PAD_SIZE; col++) {
            uint8_t NZ = black_neighbors_around(src, row, col);
            uint8_t TR_P1 = wb_transitions_around(src, row, col);
            uint8_t TR_P2 = wb_transitions_around(src, row-1, col);
            uint8_t TR_P4 = wb_transitions_around(src, row, col-1);
            uint8_t P2 = P2(src, row, col);
            uint8_t P4 = P4(src, row, col);
            uint8_t P6 = P6(src, row, col);
            uint8_t P8 = P8(src, row, col);

            uint8_t thinning_cond_1 = ((2 <= NZ) & (NZ <= 6));
            uint8_t thinning_cond_2 = (TR_P1 == 1);
            uint8_t thinning_cond_3 = (((P2 & P4 & P8) == 0) | (TR_P2 != 1));
            uint8_t thinning_cond_4 = (((P2 & P4 & P6) == 0) | (TR_P4 != 1));

            // if (thinning_cond_1 && thinning_cond_2 && thinning_cond_3 && thinning_cond_4) {
            //     dst->data[row * src->width + col] = BINARY_WHITE;
            // } else {
            //     dst->data[row * src->width + col] = src->data[row * src->width + col];
            // }

            // The code below is functionally equivalent to the if statement
            // above. This is done to avoid branching in the CUDA kernel that
            // will be based on this code.
            uint8_t thinning_ok = thinning_cond_1 & thinning_cond_2 & thinning_cond_3 & thinning_cond_4;
            uint8_t value_to_store = (src->data[row * src->width + col] * (1 - thinning_ok)) + BINARY_WHITE;
            dst->data[row * src->width + col] = value_to_store;
        }
    }
}

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

    // Pad the binary images with PAD_SIZE pixels on each side. This will be
    // useful when implementing the skeletonization algorithm, because the mask
    // we use depends on P2 and P4, which also have their own window.
    pad_binary_bitmap(&src_bitmap, BINARY_WHITE, PAD_SIZE);
    pad_binary_bitmap(&dst_bitmap, BINARY_WHITE, PAD_SIZE);

    unsigned int iterations = 0;
    do {
        skeletonize_pass(src_bitmap, dst_bitmap);
        swap_bitmaps(&src_bitmap, &dst_bitmap);

        iterations++;
    } while (!are_identical_bitmaps(src_bitmap, dst_bitmap));

    // Remove extra padding that was added to the images (don't care about
    // src_bitmap, so only need to unpad dst_bitmap)
    unpad_bitmap(&dst_bitmap, PAD_SIZE);

    // save 8-bit binary-valued grayscale version of dst_bitmap to dst_fname
    binary_to_grayscale(dst_bitmap);
    int save_successful = saveBitmap(dst_fname, dst_bitmap);
    assert(save_successful && "Could not save dst_bitmap");

    // deallocate memory used for bitmaps
    free(src_bitmap);
    free(dst_bitmap);

    return iterations;
}

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
