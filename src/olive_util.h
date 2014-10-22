/**
 * Utils
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-22
 */

#ifndef OLIVE_UTIL_H
#define OLIVE_UTIL_H


/**
 * Cuda runtime error checking. The cuda runtime errors will be treated as
 * fatal errors and will make the program exit.
 */
#define CUT_CHECK_ERROR()                                           \
    do {                                                            \                                                   \
        cudaError_t err = cudaGetLastError();                       \
        if (cudaSuccess != err) {                                   \
            fprintf(stderr, "CudaError<%s : %i> : %s.\n"),          \
                    __FILE__, __LINE__, cudaGetErrorString(err) );  \
            assert(0);                                              \
        }                                                           \
        err = cudaThreadSynchronize();                              \
        if (cudaSuccess != err) {                                   \
            fprintf(stderr, "CudaError<%s : %i> : %s.\n"),          \
                    __FILE__, __LINE__, cudaGetErrorString(err) );  \
            assert(0);                                              \
        }                                                           \ 
    } while (0);

/**
 * CUT_CALL_SAFE is used to wrap around the cuda runtime call to 
 * check if it causes any error.
 */
#define CUT_CALL_SAFE(func) { func; CUT_CHECK_ERROR(); }


// @return the number of the GPUs
int get_num_gpus(void);

// set the number of GPUs
void set_gpu_num(int);

// initialize the timer before we can use time_elapsed() 
void init_timer(void);

// @return the current time 
double get_time(void);

// @return the elapsed time between two continous call to this function
double time_elapsed(void);


// Timing info printer
#ifdef OLIVE_TIMING
    #define print_time(...) do {        \
        fprintf(stdout, "Tim: ");       \
        fprintf(stdout, __VA_ARGS__);   \
        fprintf(stdout, "\n");          \
        fflush(stdout); } while (0);    
#else
    #define print_time(...)  // disregarded
#endif

// Logging info printer
#ifdef OLIVE_LOGGING
    #define print_log(...) do {         \
        fprintf(stdout, "Log: ");       \
        fprintf(stdout, __VA_ARGS__);   \
        fprintf(stdout, "\n");          \
        fflush(stdout); } while (0);
#else
    #define print_log(...) // disregarded
#endif

// Fatal error (can not be disregarded) 
#define print_error(...) do {           \
        fprintf(stderr, "Error: ");     \
        fprintf(stderr, __VA_ARGS__);   \
        fprintf(stderr, "\n");          \
        fflush(stdout); } while (0);


#endif // OLIVE_UTIL_H


