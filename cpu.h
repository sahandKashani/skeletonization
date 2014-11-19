#ifndef CPU_H_
#define CPU_H_

#include "lspbmp.h"

#define BINARY_BLACK 1
#define BINARY_WHITE 0
#define GRAYSCALE_BLACK 0
#define GRAYSCALE_WHITE 255

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

#define NUMBER_OF_FILES 12

const char* src_file_names[NUMBER_OF_FILES] = {"../images/binary/avatar.bmp",
                                               "../images/binary/cameleon.bmp",
                                               "../images/binary/frozen.bmp",
                                               "../images/binary/giraffe.bmp",
                                               "../images/binary/snow_leopard.bmp",
                                               "../images/binary/tmnt_1.bmp",
                                               "../images/binary/tmnt_2.bmp",
                                               "../images/binary/transformers.bmp",
                                               "../images/binary/tree_1.bmp",
                                               "../images/binary/tree_2.bmp",
                                               "../images/binary/tree_3.bmp",
                                               "../images/binary/zebra.bmp"};

const char* dst_file_names[NUMBER_OF_FILES] = {"../images/skeletonized/avatar.bmp",
                                               "../images/skeletonized/cameleon.bmp",
                                               "../images/skeletonized/frozen.bmp",
                                               "../images/skeletonized/giraffe.bmp",
                                               "../images/skeletonized/snow_leopard.bmp",
                                               "../images/skeletonized/tmnt_1.bmp",
                                               "../images/skeletonized/tmnt_2.bmp",
                                               "../images/skeletonized/transformers.bmp",
                                               "../images/skeletonized/tree_1.bmp",
                                               "../images/skeletonized/tree_2.bmp",
                                               "../images/skeletonized/tree_3.bmp",
                                               "../images/skeletonized/zebra.bmp"};

typedef struct {
    uint8_t top;
    uint8_t bottom;
    uint8_t left;
    uint8_t right;
} Padding;

void print_bitmap_info(const char* fname);
void grayscale_to_binary(Bitmap* image);
void binary_to_grayscale(Bitmap* image);
void pad_binary_bitmap(Bitmap** image, uint8_t binary_padding_value, Padding padding);
void unpad_bitmap(Bitmap** image, Padding padding);
void copy_bitmap(Bitmap* src, Bitmap* dst);
void swap_bitmaps(Bitmap** src, Bitmap** dst);
void skeletonize_pass(Bitmap* src, Bitmap* dst, Padding padding);
unsigned int skeletonize(const char* src_fname, const char* dst_fname);
uint8_t is_binary_valued_grayscale_image(Bitmap* image);
uint8_t is_binary_image(Bitmap* image);
uint8_t wb_transitions_around(Bitmap* bitmap, unsigned int row, unsigned int col);
uint8_t black_neighbors_around(Bitmap* bitmap, unsigned int row, unsigned int col);
uint8_t are_identical_bitmaps(Bitmap* src, Bitmap* dst);

#endif
