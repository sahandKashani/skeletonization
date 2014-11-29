#include <assert.h>
#include <libgen.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cpu.hpp"
#include "../common/files.hpp"
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
        skeletonize_pass(src_bitmap->data, dst_bitmap->data, src_bitmap->width, src_bitmap->height, padding_amounts);
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
void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, unsigned int height, Padding padding) {
    assert(src && "src bitmap must be non-NULL");
    assert(dst && "dst bitmap must be non-NULL");

    unsigned int row = 0;
    unsigned int col = 0;
    for (row = padding.top; row < height - padding.bottom; row++) {
        for (col = padding.left; col < width - padding.right; col++) {
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

int main(void) {
    unsigned int i = 0;
    for (i = 0; i < NUMBER_OF_FILES; i++) {
        int file_name_len = strlen(src_file_names[i]);
        char* file_name = (char*) calloc(file_name_len + 1, sizeof(char));
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
