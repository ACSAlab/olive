/**
 * Defines
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */

#ifndef DEFINES_H
#define DEFINES_H

#include "common.h"

/** One word equals 64 bit. */
typedef uint64_t Word;

/** Defines type of the number space for vertex id. */
typedef uint32_t VertexId;

/** Defines the number space for edge id. */
typedef uint32_t EdgeId;

/** Defines the type for the partition identifier. */
typedef unsigned int PartitionId;

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



#endif  // DEFINES_H
