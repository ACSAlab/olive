/**
 * Unit test for the bitmap
 * 
 * Function: test the bitmap function with the specific case 
 *
 * Created by: Ye Li (mailly1994@gmail.com)
 * Created on: 2014-11-07
 * Last Modified: 2014-11-14
 */

/**/
#include <iostream>
#include <algorithm>

#include "bitmap.h"

/**/
#define DEBUG(a) std::cout <<"\033[31;1m" <<a <<"\033[0m"
#define MAXLEN 512

int get_rand(int modulo)  {
    return rand()%modulo;
}

bool bitmap_equal_array(const Bitmap& b, bool A[], int size) {
    for (int i = 0; i < size; i++) {
        if (b.get(i) != A[i]) {
            std::cerr <<i <<"get=" <<b.get(i) <<"array=" <<A[i] <<std::endl;
            return false;
        }
    }
    return true;
}

/* test operations on randomly generated bitmaps and binary_arrays */
void unitest_bitmap_operations(void) {
    int length1 = get_rand(MAXLEN);
    int length2 = get_rand(MAXLEN);
    printf("test bit operation with size %d and %d\n", length1, length2);

    Bitmap bitmap1(length1);
    Bitmap bitmap2(length2);

    bool * binary_array1 = new bool[length1]();
    bool * binary_array2 = new bool[length2]();

    // Randomly set half elements of bitmap1 and bit_array1 to 1
    for (int i = 0; i < length1/2; i++) {
        int position = get_rand(length1);
        // DEBUG(i<<":" <<position <<"\n");
        bitmap1.set(position);
        binary_array1[position] = true;
    }
    assert(bitmap_equal_array(bitmap1, binary_array1, length1));
    printf("set bitmap1\n");

/**/
    // Randomly set half elements of bitmap2 and bit_array2 to 1
    for (int i = 0; i < length2/2; i++) {
        int position = get_rand(length2);
        // DEBUG(i<<":" <<position <<"\n");
        bitmap2.set(position);
        binary_array2[position] = true;
    }
    assert(bitmap_equal_array(bitmap2, binary_array2, length2));
    printf("set bitmap2\n");

/**/
    int min = std::min(length1, length2);
    int max = std::max(length1, length2);
    Bitmap b;
    bool * a = new bool[max]();

/**/
    // &
    b = bitmap1 & bitmap2;
    for (int i = 0; i < min; i++) {
        a[i] = binary_array1[i] && binary_array2[i];
    }
    assert(bitmap_equal_array(b, a, max));
    printf("and pass\n");

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
    printf("or pass\n");

/**/
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
    printf("xor pass\n");
}

/* randomly set and unset bitmap and compare it with the binary array */
void unitest_bitmap_set_and_unset(void) {
    int length = get_rand(MAXLEN);
    printf("test set and unset with size %d\n", length);

    bool * binary_array = new bool[length]();
    for (int i = 0; i < length; i++)
        binary_array[i] = false;
    Bitmap bitmap(length);

    // Randomly sets half of the elements 1
    for (int i = 0; i < length/2; i++) {
        int position = get_rand(length);
        bitmap.set(position);
        binary_array[position] = true;
    }
    assert(bitmap_equal_array(bitmap, binary_array, length));
    printf("set pass\n");

    // Randomly sets half of the elements 0
    for (int i = 0; i < length/2; i++) {
        int position = get_rand(length);
        bitmap.unset(position);
        binary_array[position] = false;
    }
    assert(bitmap_equal_array(bitmap, binary_array, length));
    printf("unset pass\n");
}

int main(int argc, char ** arg) {
    srand(time(NULL));
    unitest_bitmap_set_and_unset();
    unitest_bitmap_operations();
    printf("Hopefully, all cases passed\n");
    return 0;
}

