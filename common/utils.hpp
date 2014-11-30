#ifndef UTILS_HPP
#define UTILS_HPP

#include <stdint.h>
#include "lspbmp.hpp"

#define BINARY_BLACK 1
#define BINARY_WHITE 0
#define GRAYSCALE_BLACK 0
#define GRAYSCALE_WHITE 255

typedef struct {
    unsigned int top;
    unsigned int bottom;
    unsigned int left;
    unsigned int right;
} Padding;

uint8_t are_identical_bitmaps(Bitmap* src, Bitmap* dst);
void binary_to_grayscale(Bitmap* image);
void copy_bitmap(Bitmap* src, Bitmap* dst);
uint8_t is_binary_image(Bitmap* image);
uint8_t is_binary_valued_grayscale_image(Bitmap* image);
void grayscale_to_binary(Bitmap* image);
void pad_binary_bitmap(Bitmap** image, uint8_t binary_padding_value, Padding padding);
void print_bitmap_info(const char* fname);
void swap_bitmaps(Bitmap** src, Bitmap** dst);
void unpad_bitmap(Bitmap** image, Padding padding);

#endif
