#ifndef UTILS_HPP
#define UTILS_HPP

#include <stdint.h>
#include "lspbmp.hpp"

#define BINARY_BLACK (1)
#define BINARY_WHITE (0)
#define GRAYSCALE_BLACK (0)
#define GRAYSCALE_WHITE (255)

uint8_t are_identical_bitmaps(Bitmap* src, Bitmap* dst);
void binary_to_grayscale(Bitmap* image);
void copy_bitmap(Bitmap* src, Bitmap* dst);
void grayscale_to_binary(Bitmap* image);
uint8_t is_binary_image(Bitmap* image);
uint8_t is_binary_valued_grayscale_image(Bitmap* image);
double percentage_black_pixels(Bitmap* image);
double percentage_white_pixels(Bitmap* image);
void print_bitmap_info(const char* fname);
void swap_bitmaps(void** src, void** dst);

#endif
