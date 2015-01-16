/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Yichao Cheng
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


/**
 * Common defines, including constants, macros and types.
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

#include "cuda_runtime.h"

/** One word equals 64 bit. */
typedef uint64_t Word;


/** Defines type of the number space for vertex id. */
typedef uint32_t VertexId;


/** Defines the number space for edge id. */
typedef uint32_t EdgeId;


/** Defines the type for the partition identifier. */
typedef uint32_t PartitionId;

/**
 * Constants for kernel configuration.
 */
const int DEFAULT_THREADS_PER_BLOCK = 256;

/**
 * Machine-specified limits.
 */
const int MAX_BLOCKS = 65535;

/**
 * Machine-specified limits. 1024 for sm3.5, 512 for sm2.0
 */
const int MAX_THREADS_PER_BLOCK = 1024;

/**
 * Machine-specified limits.
 */
const int MAX_THREADS = MAX_THREADS_PER_BLOCK * MAX_BLOCKS;


/**
 * Constants for thread identification. Only one dimension is used.
 */
#define BLOCK_INDEX blockIdx.x

#define THREAD_INDEX (threadIdx.x + blockDim.x * blockIdx.x)

/**
 * A wrapper that asserts the success of CUDA calls
 * TODO(onesuper): replace it with a method which throws an exception
 *
 */
#define CUDA_CHECK(cuda_call)                                               \
    do {                                                                    \
        cudaError_t err = cuda_call;                                        \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            assert(false);                                                  \
        }                                                                   \
    } while (0);


#define H2D(dst, src, size) cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define D2H(dst, src, size) cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)

class Managed {
public:
    void *operator new(size_t len) {
        void *ptr;
        CUDA_CHECK(cudaMallocManaged(&ptr, len));
        return ptr;
    }

    void operator delete(void *ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
};


#endif  // COMMON_H
