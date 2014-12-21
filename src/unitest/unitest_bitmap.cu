/**
 * Unit test for the bitmap
 *
 * Function: test the bitmap function with the specific case
 *
 * Created by: Ye Li (mailly1994@gmail.com)
 * Created on: 2014-11-07
 * Last Modified: 2014-11-14
 */

#include <iostream>
#include <algorithm>

#include "bitmap.h"
#include "unitest_common.h"

#define MAXLEN 512

/* test operations on randomly generated bitmaps and binary_arrays */
void unitest_bitmap_cpu() {
    int length1 = get_rand(MAXLEN);
    int length2 = get_rand(MAXLEN);
    printf("test bit operation on CPU with size %d (bitmap1) and %d (bitmap2):\n", length1, length2);

    Bitmap bitmap1(length1);
    Bitmap bitmap2(length2);
    bool *binary_array1 = new bool[length1]();
    bool *binary_array2 = new bool[length2]();

    // Randomly set half elements of bitmap1 and bit_array1 to 1
    for (int i = 0; i < length1 / 2; i++) {
        int position = get_rand(length1);
        bitmap1.set(position);
        binary_array1[position] = true;
    }
    assert(bitmap_equal_array(bitmap1, binary_array1, length1));
    printf("set bitmap1 pass\n");

    // Randomly set half elements of bitmap2 and bit_array2 to 1
    for (int i = 0; i < length2 / 2; i++) {
        int position = get_rand(length2);
        bitmap2.set(position);
        binary_array2[position] = true;
    }
    assert(bitmap_equal_array(bitmap2, binary_array2, length2));
    printf("set bitmap2 pass\n");

    int min = std::min(length1, length2);
    int max = std::max(length1, length2);
    Bitmap b;
    bool *a = new bool[max]();

    // &
    b = bitmap1 & bitmap2;
    for (int i = 0; i < min; i++) {
        a[i] = binary_array1[i] && binary_array2[i];
    }
    assert(bitmap_equal_array(b, a, max));
    printf("bitmap1 & bitmap2 pass\n");

    // |
    b = bitmap1 | bitmap2;
    for (int i = 0; i < min; i++) {
        a[i] = binary_array1[i] || binary_array2[i];
    }
    for (int i = min; i < length1; i++) {
        a[i] = binary_array1[i];
    }
    for (int i = min; i < length2; i++) {
        a[i] = binary_array2[i];
    }
    assert(bitmap_equal_array(b, a, max));
    printf("bitmap1 or bitmap2 pass\n");

    // ^
    b = bitmap1 ^ bitmap2;
    for (int i = 0; i < min; i++) {
        a[i] = binary_array1[i] != binary_array2[i];
    }
    for (int i = min; i < length1; i++) {
        a[i] = binary_array1[i];
    }
    for (int i = min; i < length2; i++) {
        a[i] = binary_array2[i];
    }
    assert(bitmap_equal_array(b, a, max));
    printf("bitmap1 ^ bitmap2 pass\n");
}

__global__
void Kernel_by_pointer(Bitmap *bitmap) {
    if (threadIdx.x % 3 == 0)
        bitmap->set(threadIdx.x);
    if (threadIdx.x % 2 == 0)
        bitmap->unset(threadIdx.x);
}

void unitest_bitmap_gpu() {
    int length = get_rand(MAXLEN);
    Bitmap *bitmap = new Bitmap(length);
    bool *binary_array = new bool[length]();
    printf("test bit operation on GPU with size %d:\n", length);

    // Randomly set half elements of bitmap to 1
    for (int i = 0; i < length / 2; i++) {
        int position = get_rand(length);
        bitmap->set(position);
        binary_array[position] = true;
    }
    // set and unset on CPU side is working
    assert(bitmap_equal_array(*bitmap, binary_array, length));
    printf("set pass\n");

    Kernel_by_pointer <<< 1, length >>>(bitmap);
    cudaDeviceSynchronize();
    for (int i = 0; i < length; i++) {
        if (i % 3 == 0)
            binary_array[i] = true;
        if (i % 2 == 0)
            binary_array[i] = false;
    }
    // the atomic set and unset on GPU is working
    assert(bitmap_equal_array(*bitmap, binary_array, length) );
    printf("set and unset pass\n");
}

int main(int argc, char **arg) {
    srand(time(NULL));
    unitest_bitmap_cpu();
    unitest_bitmap_gpu();
    printf("Hopefully, all cases passed\n");
    return 0;
}

