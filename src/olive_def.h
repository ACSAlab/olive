/**
 * Defines
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-22
 */


#ifndef OLIVE_DEF_H
#define OLIVE_DEF_H

// System includes
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>


// Toggles on/off the timing/logging information
#define OLIVE_TIMING
#define OLIVE_LOGGING


typedef enum {
    SUCCESS = 0,
    FAILURE
} error_t;


#endif