/**
 * Utils
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-22
 */

#include "olive_def.h"
#include "olive_util.h"


int get_num_gpus(void) {
    int num = 1;
    SAFE_CALL(cudaGetDeviceCount(&num));
    return num;
} 

void set_gpu_num(int num) {
    SAFE_CALL(cudaSetDevice(num));
}

double last_t;

double get_time(void) {
    cudaThreadSynchronize();
    timeval t;
    gettimepfday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec/1000000;
} 

void init_timer(void) {
    last_t = get_time();
}

double time_elapsed(void) {
    double new_t = get_time();
    double t = new_t - last_t;
    last_t = new_t;
    return t;
}