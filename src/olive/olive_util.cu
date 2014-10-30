/**
 * Utils
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */



#include "olive_util.h"


int getNumGpus(void) {
    int num = 1;
    CUT_CALL_SAFE(cudaGetDeviceCount(&num));
    return num;
}

void setGpuNum(int num) {
    CUT_CALL_SAFE(cudaSetDevice(num));
}

void checkAvailableMemory(void) {
    size_t available = 0; size_t total = 0;
    CUT_CALL_SAFE(cudaMemGetInfo(&available, &total));
    oliveLog("available memory: %llu / %llu", available, total);
}

bool isNumeric(char * str) {
  assert(str);
  while ((* str) != '\0') {
    if (!isdigit(* str)) {
        return false;
    } else {
        str++;
    }
  }
  return true;
}

double Timer::getTime(void) {
    cudaThreadSynchronize();
    timeval t;
    gettimeofday(&t, NULL);
    return static_cast<double> (t.tv_sec) +
           static_cast<double> (t.tv_usec / 1000000);
}

void Timer::initialize(void) {
    this->last_time = getTime();
}

double Timer::elapsedTime(void) {
    double new_time = getTime();
    double elapesed = new_time - this->last_time;
    this->last_time = new_time;
    return elapesed;
}



