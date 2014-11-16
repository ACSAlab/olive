/**
 * Common defines, including macros and types
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <assert.h>
#include <inttypes.h>



/**
 * A wrapper that asserts the success of cuda calls
 * TODO(onesuper): replace it with a method which throws an exception
 * 
 */
#define CUDA_CHECK(cuda_call)                                           \
    do {                                                                \
        cudaError_t err = cuda_call;                                    \
        if (err != cudaSuccess) {                                       \
        fprintf(stderr, "Cuda Error in file '%s' in line %i : %s.\n",   \
                __FILE__, __LINE__, cudaGetErrorString(err));           \
        assert(false);                                                  \
        }                                                               \
    } while (0)


/** One word equals 64 bit. */
typedef uint64_t Word;


/** Defines type of the number space for vertex id. */
typedef uint32_t VertexId;


/** Defines the number space for edge id. */
typedef uint32_t EdgeId;


/** Defines the type for the partition identifier. */
typedef uint32_t PartitionId;


/** Generic success and failure */
typedef enum {
    SUCCESS = 0,
    FAILURE,
} Error;


/** Different levels of memory location  */
typedef enum {
    CPU_ONLY = 0,   // Normal C allocations
    PINNED,         // Zero-copy memory
    MAPPED,         // Allocate on host side and maps the allocation into
                    // CUDA address space. The device pointer to the memory
                    // is obtained by calling cudaHostGetDevicePointer().
    MANAGED,        // Unified memory supported in CUDA 6.0
    GPU_ONLY,       // Normal CUDA allocations
} MemoryLevel;



#endif  // COMMON_H
