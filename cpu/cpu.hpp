#ifndef CPU_HPP
#define CPU_HPP

#include <stdint.h>
#include "../common/lspbmp.hpp"
#include "../common/utils.hpp"

uint8_t black_neighbors_around(uint8_t* data, int row, int col, unsigned int width, unsigned int height);
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap);
void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, unsigned int height);
uint8_t wb_transitions_around(uint8_t* data, int row, int col, unsigned int width, unsigned int height);

uint8_t P2_f(uint8_t* data, int row, int col, unsigned int width, unsigned int height);
uint8_t P3_f(uint8_t* data, int row, int col, unsigned int width, unsigned int height);
uint8_t P4_f(uint8_t* data, int row, int col, unsigned int width, unsigned int height);
uint8_t P5_f(uint8_t* data, int row, int col, unsigned int width, unsigned int height);
uint8_t P6_f(uint8_t* data, int row, int col, unsigned int width, unsigned int height);
uint8_t P7_f(uint8_t* data, int row, int col, unsigned int width, unsigned int height);
uint8_t P8_f(uint8_t* data, int row, int col, unsigned int width, unsigned int height);
uint8_t P9_f(uint8_t* data, int row, int col, unsigned int width, unsigned int height);

#endif
