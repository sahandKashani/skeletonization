#!/bin/sh

nvcc gpu2.cu ../common/lspbmp.cpp ../common/utils.cpp -o ../../bin/gpu2 --compiler-options "-g -O3"
