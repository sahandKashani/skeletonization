#!/bin/sh

nvcc gpu4.cu ../common/lspbmp.cpp ../common/utils.cpp -o ../../bin/gpu4 --compiler-options "-g -O3"
