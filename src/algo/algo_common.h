

#include <stdio.h>

const int INF = 0x7fffffff;


#define H2D(dst, src, size) cudaMemcpy(dst, src, sizeof(int), cudaMemcpyHostToDevice)
#define D2H(dst, src, size) cudaMemcpy(dst, src, sizeof(int), cudaMemcpyDeviceToHost)