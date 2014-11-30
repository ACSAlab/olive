#!/usr/bin/sh 
cd ..
make -f makefile_bitmap
cd ..
build/bin/unitest_bitmap

