#ifndef CPU_HPP
#define CPU_HPP

#include <stdint.h>
#include "../common/lspbmp.hpp"
#include "../common/utils.hpp"

uint8_t black_neighbors_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width);
unsigned int skeletonize(Bitmap** src_bitmap, Bitmap** dst_bitmap, Padding padding);
void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, unsigned int height, Padding padding);
uint8_t wb_transitions_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width);

#endif
