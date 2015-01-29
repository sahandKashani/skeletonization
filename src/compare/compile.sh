#!/bin/bash

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd "$DIR"

g++ -g -O3 compare.cpp ../common/lspbmp.cpp -o ../../bin/compare;
