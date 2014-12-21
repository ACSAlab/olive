/**
 * Unit test
 * 
 *
 * Created on: 2014-11-15
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include <iostream>
#include "bitmap.h"

int get_rand(int modulo)  {
    return rand() % modulo;
}

 bool bitmap_equal_array(const Bitmap &b, bool A[], int size) {
    for (int i = 0; i < size; i++) {
        if (b.get(i) != A[i]) {
            std::cerr << i << "get=" << b.get(i) << "array=" << A[i] << std::endl;
            return false;
        }
    }
    return true;
}