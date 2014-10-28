/**
 * Utils
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
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

void check_available_memory(void) {
    size_t available = 0; size_t total = 0;
    CUT_SAFE_CALL(cudaMemGetInfo(&available, &total));
    print_log("available memory: %llu / %llu", available, total);
}

inline bool is_numeric(char * str) {
  assert(str);
  while ((* str) != '\0') {
    if (!isdigit(* str)) {
        return false;
    } else {
        str++;    }
  }
  return true;
}

double Timer::get_time(void) {
    cudaThreadSynchronize();
    timeval t;
    gettimepfday(&t, NULL);
    return static<double> t.tv_sec + static<double> t.tv_usec/1000000;
}

void Timer::initialize(void) {
    this->last_time = get_time();
}

double Timer::elapsed_time(void) {
    double new_time = get_time();
    double elapesed = new_time - this->last_time;
    this->last_time = new_time;
    return elapesed;
}



