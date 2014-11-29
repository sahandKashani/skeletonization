#!/bin/sh

nvcc gpu1.cu gpu1.c ../common/lspbmp.c ../common/utils.c -o ../../bin/gpu1 --compiler-options "-g -O0"
