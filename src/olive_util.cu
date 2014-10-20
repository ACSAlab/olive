

#include "olive_util.h"


#include <cuda.h>
#include <sys/time.h>

//=============================================
// # of GPUs
//=============================================
int get_num_gpus(void) {
    int num = 1;
    SAFE_CALL(cudaGetDeviceCount(&num));
    return num;
} 

void set_gpu_num(int num) {
    SAFE_CALL(cudaSetDevice(num));
}


//=============================================
// Timer
//=============================================
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