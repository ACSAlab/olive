

#include "util/Bitmap.h"
#include "stdio.h"
#include <assert.h>


void bitmap_equal_array(const Bitmap& b, bool A[], int size) {
    for (int i = 0; i < size; i++) {
        assert(b.get(i) == A[i]);
    }
}

int unitest_bitmap_and(void) {
    Bitmap b1(3);
    b1.set(0);
    b1.unset(1);
    b1.set(2);
    bool A1[] = {true, false, true};
    bitmap_equal_array(b1, A1, 3);          // test set

    Bitmap b1_copy;
    b1_copy = b1;
    bitmap_equal_array(b1_copy, A1, 3);     // test =

    Bitmap b2(3);
    b2.set(0);
    b2.set(1);
    b2.unset(2);

    Bitmap b3;
    b3 = b1 & b2;
    bool A3[] = {true, false, false};       // test &
    bitmap_equal_array(b3, A3, 3);

    return 0;
}



int main(int argc, char ** arg) {
    unitest_bitmap_and();
    printf("Hopefully, all cases passed\n");
    return 0;
}

