#ifndef CPU_ONLY_UTILS_HPP
#define CPU_ONLY_UTILS_HPP

#include "../../common/lspbmp.hpp"
#include "../../common/utils.hpp"

void cpu_post_skeletonization(char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap);
void cpu_pre_skeletonization(int argc, char** argv, Bitmap** src_bitmap, Bitmap** dst_bitmap);

#endif
