#ifndef CPU_H
#define CPU_H

#include <stdint.h>
#include "../common/lspbmp.h"
#include "../common/utils.h"

uint8_t black_neighbors_around(Bitmap* bitmap, unsigned int row, unsigned int col);
unsigned int skeletonize(const char* src_fname, const char* dst_fname);
void skeletonize_pass(Bitmap* src, Bitmap* dst, Padding padding);
uint8_t wb_transitions_around(Bitmap* bitmap, unsigned int row, unsigned int col);

#endif
