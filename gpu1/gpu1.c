#include <assert.h>
#include <libgen.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gpu1.h"
#include "../common/files.h"
#include "../common/utils.h"

#define PAD_TOP 2
#define PAD_LEFT 2
#define PAD_BOTTOM 1
#define PAD_RIGHT 1

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
        // skeletonize_pass(src_bitmap->data, dst_bitmap->data, src_bitmap->width, src_bitmap->height, padding_amounts);
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

int main(void) {
    unsigned int i = 0;
    // for (i = 0; i < NUMBER_OF_FILES; i++) {
    for (i = 0; i < 1; i++) {
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
