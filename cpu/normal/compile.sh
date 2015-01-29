#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "$DIR"

g++ -g -O3 cpu.cpp cpu_only_utils.cpp ../../common/lspbmp.cpp ../../common/utils.cpp -o ../../../bin/cpu
