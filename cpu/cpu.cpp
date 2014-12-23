#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "cpu.hpp"
#include "../common/cpu_only_utils.hpp"
#include "../common/utils.hpp"

#define P2(data, row, col, width, height) (is_outside_image((row) - 1, (col), (width), (height)) ? BINARY_WHITE : (data)[((row) - 1) * (width) + (col)])
#define P3(data, row, col, width, height) (is_outside_image((row) - 1, (col) - 1, (width), (height)) ? BINARY_WHITE : (data)[((row) - 1) * (width) + ((col) - 1)])
#define P4(data, row, col, width, height) (is_outside_image((row), (col) - 1, (width), (height)) ? BINARY_WHITE : (data)[(row) * (width) + ((col) - 1)])
#define P5(data, row, col, width, height) (is_outside_image((row) + 1, (col) - 1, (width), (height)) ? BINARY_WHITE : (data)[((row) + 1) * (width) + ((col) - 1)])
#define P6(data, row, col, width, height) (is_outside_image((row) + 1, (col), (width), (height)) ? BINARY_WHITE : (data)[((row) + 1) * (width) + (col)])
#define P7(data, row, col, width, height) (is_outside_image((row) + 1, (col) + 1, (width), (height)) ? BINARY_WHITE : (data)[((row) + 1) * (width) + ((col) + 1)])
#define P8(data, row, col, width, height) (is_outside_image((row), (col) + 1, (width), (height)) ? BINARY_WHITE : (data)[(row) * (width) + ((col) + 1)])
#define P9(data, row, col, width, height) (is_outside_image((row) - 1, (col) + 1, (width), (height)) ? BINARY_WHITE : (data)[((row) - 1) * (width) + ((col) + 1)])

// Computes the number of black neighbors around a pixel.
uint8_t black_neighbors_around(uint8_t* data, int row, int col, unsigned int width, unsigned int height, unsigned int iterations) {
    uint8_t count = 0;

    if (row == 1348 && col == 777 && iterations == 0) {
        printf("P2 = %u\n", P2(data, row, col, width, height));
        printf("P3 = %u\n", P3(data, row, col, width, height));
        printf("P4 = %u\n", P4(data, row, col, width, height));
        printf("P5 = %u\n", P5(data, row, col, width, height));
        printf("P6 = %u\n", P6(data, row, col, width, height));
        printf("P7 = %u\n", P7(data, row, col, width, height));
        printf("P8 = %u\n", P8(data, row, col, width, height));
        printf("P9 = %u\n", P9(data, row, col, width, height));
    }

    count += (P2(data, row, col, width, height) == BINARY_BLACK);
    count += (P3(data, row, col, width, height) == BINARY_BLACK);
    count += (P4(data, row, col, width, height) == BINARY_BLACK);
    count += (P5(data, row, col, width, height) == BINARY_BLACK);
    count += (P6(data, row, col, width, height) == BINARY_BLACK);
    count += (P7(data, row, col, width, height) == BINARY_BLACK);
    count += (P8(data, row, col, width, height) == BINARY_BLACK);
    count += (P9(data, row, col, width, height) == BINARY_BLACK);

    return count;
}

// Performs an image skeletonization algorithm on the input Bitmap, and stores
// the result in the output Bitmap.
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap) {
    unsigned int iterations = 0;

    do {
        skeletonize_pass((*src_bitmap)->data, (*dst_bitmap)->data, (*src_bitmap)->width, (*src_bitmap)->height, iterations);
        swap_bitmaps((void**) src_bitmap, (void**) dst_bitmap);

        iterations++;
        printf(".");
        fflush(stdout);
    } while (!are_identical_bitmaps(*src_bitmap, *dst_bitmap));

    return iterations;
}

// Performs 1 iteration of the thinning algorithm.
void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, unsigned int height, unsigned int iterations) {
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            // Optimization for CPU algorithm: You don't need to do any of these
            // computations if the pixel is already BINARY_WHITE
            if (src[row * width + col] == BINARY_BLACK) {
                uint8_t NZ = black_neighbors_around(src, row, col, width, height, iterations);
                uint8_t TR_P1 = wb_transitions_around(src, row, col, width, height, iterations);
                uint8_t TR_P2 = wb_transitions_around(src, row - 1, col, width, height, iterations);
                uint8_t TR_P4 = wb_transitions_around(src, row, col - 1, width, height, iterations);
                uint8_t P2 = P2(src, row, col, width, height);
                uint8_t P4 = P4(src, row, col, width, height);
                uint8_t P6 = P6(src, row, col, width, height);
                uint8_t P8 = P8(src, row, col, width, height);

                uint8_t thinning_cond_1 = ((2 <= NZ) && (NZ <= 6));
                uint8_t thinning_cond_2 = (TR_P1 == 1);
                uint8_t thinning_cond_3 = (((P2 && P4 && P8) == 0) || (TR_P2 != 1));
                uint8_t thinning_cond_4 = (((P2 && P4 && P6) == 0) || (TR_P4 != 1));

                if (row == 1348 && col == 777 && iterations == 0) {
                    printf("======================\n");
                    printf("src[%u * %u + %u] = %u\n", row, width, col, src[row * width + col]);
                    printf("NZ = %u\n", NZ);
                    printf("TR_P1 = %u\n", TR_P1);
                    printf("TR_P2 = %u\n", TR_P2);
                    printf("TR_P4 = %u\n", TR_P4);
                    printf("P2 = %u\n", P2);
                    printf("P4 = %u\n", P4);
                    printf("P6 = %u\n", P6);
                    printf("P8 = %u\n", P8);
                    printf("thinning_cond_1 = %u\n", thinning_cond_1);
                    printf("thinning_cond_2 = %u\n", thinning_cond_2);
                    printf("thinning_cond_3 = %u\n", thinning_cond_3);
                    printf("thinning_cond_4 = %u\n", thinning_cond_4);
                    printf("======================\n");
                }

                if (thinning_cond_1 && thinning_cond_2 && thinning_cond_3 && thinning_cond_4) {
                    dst[row * width + col] = BINARY_WHITE;
                } else {
                    dst[row * width + col] = src[row * width + col];
                }
            } else {
                dst[row * width + col] = src[row * width + col];
            }
        }
    }
}

// Computes the number of white to black transitions around a pixel.
uint8_t wb_transitions_around(uint8_t* data, int row, int col, unsigned int width, unsigned int height, unsigned int iterations) {
    uint8_t count = 0;

    count += ((P2(data, row, col, width, height) == BINARY_WHITE) && (P3(data, row, col, width, height) == BINARY_BLACK));
    count += ((P3(data, row, col, width, height) == BINARY_WHITE) && (P4(data, row, col, width, height) == BINARY_BLACK));
    count += ((P4(data, row, col, width, height) == BINARY_WHITE) && (P5(data, row, col, width, height) == BINARY_BLACK));
    count += ((P5(data, row, col, width, height) == BINARY_WHITE) && (P6(data, row, col, width, height) == BINARY_BLACK));
    count += ((P6(data, row, col, width, height) == BINARY_WHITE) && (P7(data, row, col, width, height) == BINARY_BLACK));
    count += ((P7(data, row, col, width, height) == BINARY_WHITE) && (P8(data, row, col, width, height) == BINARY_BLACK));
    count += ((P8(data, row, col, width, height) == BINARY_WHITE) && (P9(data, row, col, width, height) == BINARY_BLACK));
    count += ((P9(data, row, col, width, height) == BINARY_WHITE) && (P2(data, row, col, width, height) == BINARY_BLACK));

    return count;
}

int main(int argc, char** argv) {
    Bitmap* src_bitmap = NULL;
    Bitmap* dst_bitmap = NULL;

    cpu_pre_skeletonization(argc, argv, &src_bitmap, &dst_bitmap);

    unsigned int iterations = skeletonize(&src_bitmap, &dst_bitmap);
    printf(" %u iterations\n", iterations);

    cpu_post_skeletonization(argv, &src_bitmap, &dst_bitmap);

    return EXIT_SUCCESS;
}
