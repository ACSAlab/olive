/**
 * Unit Test of util/Bitmap.h
 * 
 * Function: test the bitmap function with the specific case 
 *
 * Modified by: Ye Li (mailly1994@gmail.com)
 * Created on: 2014-11-07
 * Last Modified: 2014-11-08
 */

#include "util/Bitmap.h"
#include "stdio.h"
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>

#define DEBUG(a) std::cout<<"\033[31;1m"<<a<<"\033[0m"
#define MAXLEN 512

int get_rand(int modulo)  {
    return rand()%modulo;
}
void bitmap_equal_array(const Bitmap& b, bool A[], int size) {
    for (int i = 0; i < size; i++) {
        assert(b.get(i) == A[i] || !(std::cerr<<i<<"get="<<b.get(i)<<"array="<<A[i]<<std::endl));
    }
}

/* test operations on randomly generated bitmaps and binary_arrays */
int unitest_bitmap_operations(void) {
    DEBUG("test operations\n");
    int length1 = get_rand(MAXLEN);
    DEBUG(length1<<"\n");
    bool * binary_array = new bool[length1]();
    for (int i=0; i<length1; i++)
        binary_array[i] = false;
    Bitmap bitmap(length1);
    
    bitmap_equal_array(bitmap, binary_array, length1);
    // set
    for (int i=0; i<length1/2; i++) {
        int position = get_rand(length1);
        // DEBUG(i<<":"<<position<<"\n");
        bitmap.set(position);
        binary_array[position] = true;
        bitmap_equal_array(bitmap, binary_array, length1);
    }


    int length2 = get_rand(MAXLEN);
    DEBUG(length2<<"\n");
    bool * binary_array2 = new bool[length2]();
    for (int i=0; i<length2; i++)
        binary_array2[i] = false;
    Bitmap bitmap2(length2);
    
    bitmap_equal_array(bitmap2, binary_array2, length2);
    DEBUG("set start\n");
    // set
    for (int i=0; i<length2/2; i++) {
        int position = get_rand(length2);
        // DEBUG(i<<":"<<position<<"\n");
        bitmap2.set(position);
        binary_array2[position] = true;
        bitmap_equal_array(bitmap2, binary_array2, length2);
    }

    int min = MIN(length1, length2);
    int max = MAX(length1, length2);
    // &
    Bitmap b;
    b = bitmap & bitmap2;
    bool * a = new bool[max]();
    for (int i=0; i<min; i++) {
        a[i] = binary_array[i] && binary_array2[i];
    }
    for (int i=min; i<max; i++) {
        a[i] = false;
    }
    bitmap_equal_array(b, a, max);
    DEBUG("& passed\n");
    // |
    b = bitmap | bitmap2;
    for (int i=0; i<min; i++) {
        a[i] = binary_array[i] || binary_array2[i];
    }
    for (int i=min; i<length1; i++) {
        a[i] = binary_array[i];
    }
    for (int i=min; i<length2; i++) {
        a[i] = binary_array2[i];
    }
    bitmap_equal_array(b, a, max);
    DEBUG("| passed\n");
    // ^
    b = bitmap ^ bitmap2;
    for (int i=0; i<min; i++) {
        a[i] = binary_array[i] != binary_array2[i];
    }
    for (int i=min; i<length1; i++) {
        a[i] = binary_array[i];
    }
    for (int i=min; i<length2; i++) {
        a[i] = binary_array2[i];
    }
    bitmap_equal_array(b, a, max);
    DEBUG("^ passed\n");

    return 0;
}
/* randomly set and unset bitmap and compare it with the binary array */
int unitest_bitmap_set_and_unset(void) {
    int length = get_rand(MAXLEN);
    DEBUG(length<<"\n");
    bool * binary_array = new bool[length]();
    Bitmap bitmap(length);
    
    bitmap_equal_array(bitmap, binary_array, length);
    // set
    for (int i=0; i<length/2; i++) {
        int position = get_rand(length);
        // DEBUG(i<<":"<<position<<"\n");
        bitmap.set(position);
        binary_array[position] = true;
        bitmap_equal_array(bitmap, binary_array, length);
    }
    DEBUG("set passed\n");
    // unset
    for (int i=0; i<length/2; i++) {
        int position = get_rand(length);
        bitmap.unset(position);
        binary_array[position] = false;
    }
    bitmap_equal_array(bitmap, binary_array, length);
    DEBUG("unset passed\n");
    return 0;
}



int main(int argc, char ** arg) {
    srand(time(NULL));
    unitest_bitmap_set_and_unset();
    unitest_bitmap_operations();
    printf("Hopefully, all cases passed\n");
    return 0;
}

