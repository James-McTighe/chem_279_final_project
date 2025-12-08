#!/bin/sh
./clean.sh

mkdir -p build
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j4

cd ..
build/ir_spec MolJSON/H2.json