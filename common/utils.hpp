#ifndef UTILS_HPP
#define UTILS_HPP

#include <stdint.h>
#include "lspbmp.hpp"

#define BINARY_BLACK (1)
#define BINARY_WHITE (0)
#define GRAYSCALE_BLACK (0)
#define GRAYSCALE_WHITE (255)

#define is_outside_image(row, col, width, height) ((((int) (row)) < ((int) 0)) || (((int) (row)) > ((int) ((height) - 1))) || (((int) (col)) < ((int) 0)) || (((int) (col)) > ((int) ((width) - 1))))

uint8_t are_identical_bitmaps(Bitmap* src, Bitmap* dst);
void binary_to_grayscale(Bitmap* image);
void copy_bitmap(Bitmap* src, Bitmap* dst);
void grayscale_to_binary(Bitmap* image);
uint8_t is_binary_image(Bitmap* image);
uint8_t is_binary_valued_grayscale_image(Bitmap* image);
void print_bitmap_info(const char* fname);
void swap_bitmaps(void** src, void** dst);

#endif
