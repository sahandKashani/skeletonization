#!/bin/sh

g++ -g -O3 cpu.cpp ../common/cpu_only_utils.cpp ../common/lspbmp.cpp ../common/utils.cpp -o ../../bin/cpu;
