/**
 * Unitest for bitmap (olive/bitmap.cuh)
 *
 * Created on 2014-11-29
 * Last modified on 2014-11-30
 */

#include "bitmap.cuh"
#include <iostream>

#define SIZE 500

__global__
void Kernel_by_pointer(Bitmap *bitmap) {
	// the read access can be done concurrently
	if (bitmap->get(49) != true)
		return;
	if (bitmap->capacity() != 512)
		return;

	if (threadIdx.x %3 == 0)
		bitmap->set_gpu(threadIdx.x);
	if (threadIdx.x %2 == 0)
		bitmap->unset_gpu(threadIdx.x);
}

bool bitmap_equal_array(const Bitmap& b, bool A[], int size) {
	int i;
    for (i = 0; i < size; i++) {
        if (b.get(i) != A[i]) {
            std::cerr <<i <<"get=" <<b.get(i) <<"array=" <<A[i] <<std::endl;
            return false;
        }
    }
    return true;
}

int main() {
	Bitmap *bitmap = new Bitmap(SIZE);
	bool *binary_array = new bool[SIZE]();

	bitmap->set_cpu(299);
	bitmap->set_cpu(49);
	bitmap->set_cpu(4);
	bitmap->set_cpu(300);

	binary_array[299] = true;
	binary_array[49] = true;
	binary_array[4] = true;
	binary_array[300] = true;

	// set and unset on cpu side is working
	assert (bitmap_equal_array(*bitmap, binary_array, SIZE) );

	Kernel_by_pointer<<< 1, SIZE >>>(bitmap);
	cudaDeviceSynchronize();

	int i;
	for (i=0; i<SIZE; i++) {
		if (i % 3 == 0)
			binary_array[i] = true;
		if (i % 2 == 0)
			binary_array[i] = false;
	}
	
	// the atomic set and unset on gpu side is working
	assert (bitmap_equal_array(*bitmap, binary_array, SIZE) );
	printf("TEST PASSED!\n");

}