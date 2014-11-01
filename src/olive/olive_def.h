/**
 * Defines
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */


#ifndef OLIVE_DEF_H
#define OLIVE_DEF_H

// System includes
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>

// Toggles on/off the timing/logging information
#define OLIVE_TIMING
#define OLIVE_LOGGING

// Generic success and failure
typedef enum {
    SUCCESS = 0,
    FAILURE
} Error;

/**
 * Simulates simple exceptions: if the statement is not correct, jump to
 * label, typically an error label where you could handle the 'exception'
 */
#define TRY(stmt, label)                \
    do {                                \
        if (!(stmt))                    \
        goto label;                     \
    } while (0);

/**
 * Cuda runtime error checking. The cuda runtime errors will be treated as
 * fatal errors and will make the program exit.
 */
#define CUT_CHECK_ERROR()                                           \
    do {                                                            \
        cudaError_t err = cudaGetLastError();                       \
        if (cudaSuccess != err) {                                   \
            fprintf(stderr, "CudaError<%s : %i> : %s.\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err) );  \
            assert(0);                                              \
        }                                                           \
        err = cudaThreadSynchronize();                              \
        if (cudaSuccess != err) {                                   \
            fprintf(stderr, "CudaError<%s : %i> : %s.\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err) );  \
            assert(0);                                              \
        }                                                           \
    } while (0);

/**
 * Wraps around the cuda runtime call to check if it causes any error.
 */
#define CUT_CALL_SAFE(func_call)     \
    do {                             \
        func_call;                   \
        CUT_CHECK_ERROR();           \
    } while (0);                     \

/**
 * Timing info printer. Can be complied to no-op if wanted.
 */
#ifdef OLIVE_TIMING
    #define oliveTim(...)                       \
        do {                                    \
            fprintf(stdout, "[TIM] ");          \
            fprintf(stdout, __VA_ARGS__);       \
            fprintf(stdout, "\n");              \
            fflush(stdout);                     \
        } while (0);
#else
    #define oliveTim(...)  // no-op
#endif

/**
 * Logging info printer. Can be complied to no-op if wanted.
 */
#ifdef OLIVE_LOGGING
    #define oliveLog(...)                       \
        do {                                    \
            fprintf(stdout, "[LOG] ");          \
            fprintf(stdout, __VA_ARGS__);       \
            fprintf(stdout, "\n");              \
            fflush(stdout);                     \
        } while (0);
#else
    #define oliveLog(...)  // no-op
#endif

/**
 * Fatal errors
 * Print the message and quit program immediately.
 */
#define oliveFatal(...)                         \
    do {                                        \
        fprintf(stderr, "[FATAL] ");            \
        fprintf(stderr, __VA_ARGS__);           \
        fprintf(stderr, "\n");                  \
        fflush(stdout);                         \
        assert(false);                          \
    } while (0);

/**
 * Ordinary errors
 * Print the message and let the program handle it.
 */
#define oliveError(...)                         \
    do {                                        \
        fprintf(stderr, "Error: ");             \
        fprintf(stderr, __VA_ARGS__);           \
        fprintf(stderr, "\n");                  \
        fflush(stdout);                         \
    } while (0);


#endif
