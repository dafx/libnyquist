#!/bin/sh
echo "build..."
cmake --workflow --preset gcc-cuda
echo "run the examples"
./out/build/gcc-cuda/bin/libnyquist-examples test_data/sb-reverie.opus
