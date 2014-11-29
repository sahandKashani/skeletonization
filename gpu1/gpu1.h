#ifndef GPU1_H
#define GPU1_H

#include <stdint.h>
#include "../common/lspbmp.h"
#include "../common/utils.h"

// __device__ uint8_t black_neighbors_around(Bitmap* bitmap, unsigned int row, unsigned int col);
unsigned int skeletonize(const char* src_fname, const char* dst_fname);
// __global__ void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, unsigned int height, Padding padding);
// __device__ uint8_t wb_transitions_around(Bitmap* bitmap, unsigned int row, unsigned int col);

#endif
