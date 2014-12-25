#!/bin/sh

nvcc \
--ptxas-options=-v \
-gencode=arch=compute_10,code=sm_10 \
-gencode=arch=compute_20,code=sm_20 \
-gencode=arch=compute_30,code=sm_30 \
-gencode=arch=compute_35,code=sm_35 \
-gencode=arch=compute_35,code=compute_35 \
gpu2.cu ../common/gpu_only_utils.cu ../common/lspbmp.cpp ../common/utils.cpp \
-O3 \
-o ../../bin/gpu2 \
--compiler-options "-g -O3"
