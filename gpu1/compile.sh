#!/bin/sh

nvcc gpu1.cu ../common/lspbmp.cpp ../common/utils.cpp -o ../../bin/gpu1 --compiler-options "-g -O3"
