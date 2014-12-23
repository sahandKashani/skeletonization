#!/bin/sh

nvcc --ptxas-options=-v -arch=sm_20 gpu2.cu ../common/gpu_only_utils.cu ../common/lspbmp.cpp ../common/utils.cpp -o ../../bin/gpu2 --compiler-options "-g -O3"
