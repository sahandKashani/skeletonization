#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "$DIR"

nvcc \
--generate-line-info \
--ptxas-options=-v \
-gencode=arch=compute_20,code=sm_20 \
-gencode=arch=compute_30,code=sm_30 \
-gencode=arch=compute_35,code=sm_35 \
-gencode=arch=compute_35,code=compute_35 \
gpu1.cu ../common/gpu_only_utils.cu ../../../common/lspbmp.cpp ../../../common/utils.cpp \
-O3 \
-o ../../../../bin/gpu1_normal \
--compiler-options "-g -O3"
