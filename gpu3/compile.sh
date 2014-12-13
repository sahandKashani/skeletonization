#!/bin/sh

nvcc gpu3.cu ../common/lspbmp.cpp ../common/utils.cpp -o ../../bin/gpu3 --compiler-options "-g -O3"
