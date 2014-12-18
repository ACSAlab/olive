/**
 * Common defines
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-12-17
 * Last Modified: 2014-12-18
 */

#include <stdio.h>
#include <assert.h>

const int INF = 0x7fffffff;


#define H2D(dst, src, size) cudaMemcpy(dst, src, sizeof(int), cudaMemcpyHostToDevice)
#define D2H(dst, src, size) cudaMemcpy(dst, src, sizeof(int), cudaMemcpyDeviceToHost)


void expect_equal(std::vector<int> v1, std::vector<int> v2) {
    assert(v1.size() == v2.size());
    for (int i = 0; i < v1.size(); i++) {
        assert(v1[i] == v2[i]);
    }
}