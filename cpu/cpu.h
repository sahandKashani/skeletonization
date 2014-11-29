#ifndef CPU_H
#define CPU_H

#include <stdint.h>
#include "../common/lspbmp.h"
#include "../common/utils.h"

uint8_t black_neighbors_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width);
unsigned int skeletonize(const char* src_fname, const char* dst_fname);
void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, unsigned int height, Padding padding);
uint8_t wb_transitions_around(uint8_t* data, unsigned int row, unsigned int col, unsigned int width);

#endif
