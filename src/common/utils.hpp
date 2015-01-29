#ifndef UTILS_HPP
#define UTILS_HPP

#include <stdint.h>
#include "lspbmp.hpp"

#define BINARY_BLACK (1)
#define BINARY_WHITE (0)
#define GRAYSCALE_BLACK (0)
#define GRAYSCALE_WHITE (255)

#define PAD_TOP (2)
#define PAD_LEFT (2)
#define PAD_BOTTOM (1)
#define PAD_RIGHT (1)

typedef struct {
    int top;
    int left;
    int bottom;
    int right;
} Padding;

uint8_t are_identical_bitmaps(Bitmap* src, Bitmap* dst);
void binary_to_grayscale(Bitmap* image);
void copy_bitmap(Bitmap* src, Bitmap* dst);
void grayscale_to_binary(Bitmap* image);
uint8_t is_binary_image(Bitmap* image);
uint8_t is_binary_valued_grayscale_image(Bitmap* image);
void pad_binary_bitmap(Bitmap** image, uint8_t binary_padding_value, Padding padding);
double percentage_black_pixels(Bitmap* image);
double percentage_white_pixels(Bitmap* image);
void print_bitmap(Bitmap* image);
void print_bitmap_info(const char* fname);
void swap_bitmaps(void** src, void** dst);
void unpad_binary_bitmap(Bitmap** image, Padding padding);

#endif
