#include <assert.h>
#include <libgen.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpu.h"
#include "files.h"
#include "utils.h"

#define PAD_TOP 2
#define PAD_LEFT 2
#define PAD_BOTTOM 1
#define PAD_RIGHT 1

#define P2(bitmap, row, col) ((bitmap)->data[((row)-1) * (bitmap)->width +  (col)   ])
#define P3(bitmap, row, col) ((bitmap)->data[((row)-1) * (bitmap)->width + ((col)-1)])
#define P4(bitmap, row, col) ((bitmap)->data[ (row)    * (bitmap)->width + ((col)-1)])
#define P5(bitmap, row, col) ((bitmap)->data[((row)+1) * (bitmap)->width + ((col)-1)])
#define P6(bitmap, row, col) ((bitmap)->data[((row)+1) * (bitmap)->width +  (col)   ])
#define P7(bitmap, row, col) ((bitmap)->data[((row)+1) * (bitmap)->width + ((col)+1)])
#define P8(bitmap, row, col) ((bitmap)->data[ (row)    * (bitmap)->width + ((col)+1)])
#define P9(bitmap, row, col) ((bitmap)->data[((row)-1) * (bitmap)->width + ((col)+1)])

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
        printf(".");
        fflush(stdout);
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

int main(void) {
//    unsigned int image_index = 0;
//    skeletonize(src_file_names[image_index], dst_file_names[image_index]);

    for (unsigned int i = 0; i < NUMBER_OF_FILES; i++) {
        int file_name_len = strlen(src_file_names[i]);
        char* file_name = calloc(file_name_len + 1, sizeof(char));
        strncpy(file_name, src_file_names[i], file_name_len);
        char* base_file_name = basename(file_name);
        printf("%s ", base_file_name);
        fflush(stdout);
        free(file_name);

        unsigned int iterations = skeletonize(src_file_names[i], dst_file_names[i]);

        printf(" %d iterations\n", iterations);
        fflush(stdout);
    }

    printf("done\n");
    fflush(stdout);

    return 0;
}
