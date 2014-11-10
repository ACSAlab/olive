#!/usr/bin/sh 
cd ..
make -f makefile.unitest_bitmap
cd ..
build/bin/unitest_bitmap

while getopts "c" opt; do
        case "$opt" in    
        c)  rm build/bin/*
            ;;
        esac
done
