/**
 * Defines
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */


#pragma once

#include <assert.h>
#include <stdio.h>

/**
 * Toggles on/off the logging information
 */
#define OLIVE_TIMING
#define OLIVE_LOGGING

/**
 * Generic success and failure
 */
typedef enum {
    SUCCESS = 0,
    FAILURE,
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
#define oliveError(...)                                     \
    do {                                                    \
        fprintf(stderr, "[ERROR] ");                        \
        fprintf(stderr, __VA_ARGS__);                       \
        fprintf(stderr, ": %s (%d)", __FILE__, __LINE__);   \
        fprintf(stderr, "\n");                              \
        fflush(stdout);                                     \
    } while (0);

/**
 * Cuda Runtime errors
 * Convert a cuda error to a string and print it 
 */
#define oliveCudaError(err)                                 \
    do {                                                    \
        fprintf(stderr, "[CUERR] ");                        \
        fprintf(stderr, cudaGetErrorString(err));           \
        fprintf(stderr, ": %s (%d)", __FILE__, __LINE__);   \
        fprintf(stderr, "\n");                              \
        fflush(stdout);                                     \
    } while (0);

