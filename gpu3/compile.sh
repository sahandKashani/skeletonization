#!/bin/sh

nvcc --ptxas-options=-v gpu3.cu ../common/gpu_only_utils.cu ../common/lspbmp.cpp ../common/utils.cpp -o ../../bin/gpu3 --compiler-options "-g -O3"
