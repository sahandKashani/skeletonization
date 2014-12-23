#!/bin/sh

nvcc --ptxas-options=-v -arch=sm_20 gpu4.cu ../common/gpu_only_utils.cu ../common/lspbmp.cpp ../common/utils.cpp -o ../../bin/gpu4 --compiler-options "-g -O3"
