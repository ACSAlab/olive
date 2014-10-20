

#ifndef OLIVE_UTIL_H
#define OLIVE_UTIL_H

#include <stdio.h>

#define OLIVE_TIMING
#define OLIVE_LOGGING

//=============================================
// Cuda error checking 
//=============================================
#define CUT_CHECK_ERROR() do {                                      \                                                   \
        cudaError_t err = cudaGetLastError();                       \
        if (cudaSuccess != err) {                                   \
            fprintf(stderr, "CudaError<%s : %i> : %s.\n"),          \
                    __FILE__, __LINE__, cudaGetErrorString(err) );  \
            exit(FAILURE);                                          \
        }                                                           \
        err = cudaThreadSynchronize();                              \
        if (cudaSuccess != err) {                                   \
            fprintf(stderr, "CudaError<%s : %i> : %s.\n"),          \
                    __FILE__, __LINE__, cudaGetErrorString(err) );  \
            exit(FAILURE);                                          \
        }                                                           \ 
    } while (0);


#define SAFE_CALL(call) { call; CUT_CHECK_ERROR(); }


//=============================================
// # of GPUs
//=============================================
int get_num_gpus(void);
void set_gpu_num(int);


//=============================================
// Timer
//=============================================
double get_time(void);
void init_timer(void);
// return the elapsed time between two continous invovations
double time_elapsed(void);


//=============================================
// Logging and Error printing
//=============================================
// Timer 
#ifdef OLIVE_TIMING
    #define print_time(...) printf(__VA_ARGS__)
#else
    #define print_time(...)
#endif

// Logging
#ifdef OLIVE_LOGGING
    #define print_log(...) printf(__VA_ARGS__)
#else
    #define print_log(...)
#endif

// Fatal error
#define print_error(...) { printf(__VA_ARGS__); exit(FAILURE); }




#endif // HG