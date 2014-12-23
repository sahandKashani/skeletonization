#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/lspbmp.hpp"

int main(int argc, char** argv) {
    Bitmap* src_bitmap = NULL;
    Bitmap* dst_bitmap = NULL;

    char* src_fname = argv[1];
    char* dst_fname = argv[2];

    src_bitmap = loadBitmap(src_fname);
    assert(src_bitmap != NULL && "Error: could not load src_bitmap");

    dst_bitmap = loadBitmap(dst_fname);
    assert(dst_bitmap != NULL && "Error: could not load dst_bitmap");

    assert(src_bitmap->width == dst_bitmap->width);
    assert(src_bitmap->height == dst_bitmap->height);
    assert(src_bitmap->depth == dst_bitmap->depth);

    for (int row = 0; row < src_bitmap->height; row++) {
        for (int col = 0; col < src_bitmap->width; col++) {
            if (src_bitmap->data[row * (src_bitmap->width) + col] != dst_bitmap->data[row * (src_bitmap->width) + col]) {
                printf("mismatch : row = %03u, col = %03u\n", row, col);
            }
        }
    }

    return EXIT_SUCCESS;
}
