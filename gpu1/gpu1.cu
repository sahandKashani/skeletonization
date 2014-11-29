// #include <stdint.h>
// #include "gpu1.h"
// #include "../common/utils.h"

// #define P2(bitmap, row, col) ((bitmap)->data[((row)-1) * (bitmap)->width +  (col)   ])
// #define P3(bitmap, row, col) ((bitmap)->data[((row)-1) * (bitmap)->width + ((col)-1)])
// #define P4(bitmap, row, col) ((bitmap)->data[ (row)    * (bitmap)->width + ((col)-1)])
// #define P5(bitmap, row, col) ((bitmap)->data[((row)+1) * (bitmap)->width + ((col)-1)])
// #define P6(bitmap, row, col) ((bitmap)->data[((row)+1) * (bitmap)->width +  (col)   ])
// #define P7(bitmap, row, col) ((bitmap)->data[((row)+1) * (bitmap)->width + ((col)+1)])
// #define P8(bitmap, row, col) ((bitmap)->data[ (row)    * (bitmap)->width + ((col)+1)])
// #define P9(bitmap, row, col) ((bitmap)->data[((row)-1) * (bitmap)->width + ((col)+1)])

// // Computes the number of black neighbors around a pixel.
// __device__ uint8_t black_neighbors_around(Bitmap* bitmap, unsigned int row, unsigned int col) {
//     uint8_t count = 0;

//     count += (P2(bitmap, row, col) == BINARY_BLACK);
//     count += (P3(bitmap, row, col) == BINARY_BLACK);
//     count += (P4(bitmap, row, col) == BINARY_BLACK);
//     count += (P5(bitmap, row, col) == BINARY_BLACK);
//     count += (P6(bitmap, row, col) == BINARY_BLACK);
//     count += (P7(bitmap, row, col) == BINARY_BLACK);
//     count += (P8(bitmap, row, col) == BINARY_BLACK);
//     count += (P9(bitmap, row, col) == BINARY_BLACK);

//     return count;
// }

// // Performs 1 iteration of the thinning algorithm.
// __global__ void skeletonize_pass(uint8_t* src, uint8_t* dst, unsigned int width, unsigned int height, Padding padding) {
//     unsigned int row = 0;
//     unsigned int col = 0;
//     for (row = padding.top; row < height - padding.bottom; row++) {
//         for (col = padding.left; col < width - padding.right; col++) {
//             uint8_t NZ = black_neighbors_around(src, row, col);
//             uint8_t TR_P1 = wb_transitions_around(src, row, col);
//             uint8_t TR_P2 = wb_transitions_around(src, row-1, col);
//             uint8_t TR_P4 = wb_transitions_around(src, row, col-1);
//             uint8_t P2 = P2(src, row, col);
//             uint8_t P4 = P4(src, row, col);
//             uint8_t P6 = P6(src, row, col);
//             uint8_t P8 = P8(src, row, col);

//             uint8_t thinning_cond_1 = ((2 <= NZ) && (NZ <= 6));
//             uint8_t thinning_cond_2 = (TR_P1 == 1);
//             uint8_t thinning_cond_3 = (((P2 && P4 && P8) == 0) || (TR_P2 != 1));
//             uint8_t thinning_cond_4 = (((P2 && P4 && P6) == 0) || (TR_P4 != 1));

//             if (thinning_cond_1 && thinning_cond_2 && thinning_cond_3 && thinning_cond_4) {
//                 dst[row * width + col] = BINARY_WHITE;
//             } else {
//                 dst[row * width + col] = src[row * width + col];
//             }
//         }
//     }
// }

// // Computes the number of white to black transitions around a pixel.
// __device__ uint8_t wb_transitions_around(Bitmap* bitmap, unsigned int row, unsigned int col) {
//     uint8_t count = 0;

//     count += ( (P2(bitmap, row, col) == BINARY_WHITE) && (P3(bitmap, row, col) == BINARY_BLACK) );
//     count += ( (P3(bitmap, row, col) == BINARY_WHITE) && (P4(bitmap, row, col) == BINARY_BLACK) );
//     count += ( (P4(bitmap, row, col) == BINARY_WHITE) && (P5(bitmap, row, col) == BINARY_BLACK) );
//     count += ( (P5(bitmap, row, col) == BINARY_WHITE) && (P6(bitmap, row, col) == BINARY_BLACK) );
//     count += ( (P6(bitmap, row, col) == BINARY_WHITE) && (P7(bitmap, row, col) == BINARY_BLACK) );
//     count += ( (P7(bitmap, row, col) == BINARY_WHITE) && (P8(bitmap, row, col) == BINARY_BLACK) );
//     count += ( (P8(bitmap, row, col) == BINARY_WHITE) && (P9(bitmap, row, col) == BINARY_BLACK) );
//     count += ( (P9(bitmap, row, col) == BINARY_WHITE) && (P2(bitmap, row, col) == BINARY_BLACK) );

//     return count;
// }
