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
 * GPU-resident Dataset.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-13
 * Last Modified: 2014-11-13
 */

#ifndef GRD_H
#define GRD_H

#include "common.h"

/**
 * GPU-Resident Dataset (GRD) provides the utility for allocating data buffers
 * which can be transferred between CPU and GPU and accessed from both CPU and GPU.
 */
template<typename T>
class GRD {
public:
    T      *elemsHost;    /** Points to the host-allocated buffer */
    T      *elemsDevice;  /** Points to the GPU-allocated buffer */
    size_t  length;       /** The length of the buffer */
    int     deviceId;     /** The device GRD locates at */

    /** List Initializer */
    GRD(): elemsHost(NULL), elemsDevice(NULL), length(0), deviceId(-1) {}

    /**
     * Overloads the subscript to access an element on host side.
     * Do not check the boundaries for speed.
     */
    __host__ __device__
    inline T &operator[] (size_t index) const {
#ifdef __CUDA_ARCH__
        return elemsDevice[index];
#else
        return elemsHost[index];
#endif
    }

    /** Returning the capacity of GRD */
    inline size_t capacity() const {
        return length;
    }

    /**
     * Allocate the host- and device-resident buffer of length `len` on device
     * `id`.
     */
    inline void reserve(size_t len, int id = 0) {
        assert(len > 0);
        assert(id >= 0);
        deviceId = id;
        length = len;
        elemsHost = reinterpret_cast<T *>(malloc(len * sizeof(T)));
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&elemsDevice),
                              len * sizeof(T)));
    }

    /**
     * Set all the elements to the value `x` on the device.
     */
    void allTo(T x) {
        for (size_t i = 0; i < length; i++) {
            elemsHost[i] = x;
        }
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              length * sizeof(T), cudaMemcpyDefault));
    }

    /**
     * Clear the every bytesto of the GRD.
     */
    void clear() {
        CUDA_CHECK(cudaMemset(elemsDevice, 0, sizeof(T) * length));
    }

    /**
     * Set elements[i] to the value `x` on the host.
     */
    void set(size_t i, T x) {
        elemsHost[i] = x;
        CUDA_CHECK(cudaMemcpy(elemsDevice + i, elemsHost + i,
                              1 * sizeof(T), cudaMemcpyDefault));
    }

    /**
     * Write-backs the dataset from GPU's on-board memory to host memory.
     */
    inline void persist() {
        if (length == 0) return;
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMemcpy(elemsHost, elemsDevice,
                              length * sizeof(T), cudaMemcpyDefault));
    }


    /**
     * Call the user defined print() function to print the GRD.
     */
    inline void print() {
        for (int i = 0; i < length; i++) {
            elemsHost[i].print();
        }
    }

    inline void peek() {
        persist();
        for (int i = 0; i < length; i++) {
            std::cout << elemsHost[i] << " ";
        }
        printf("\n");
    }

    /**
     * Caches the dataset in GPU's on-board memory.
     */
    inline void cache() {
        if (length == 0) return;
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              length * sizeof(T), cudaMemcpyDefault));
    }

    /**
     * Free both host- and device- resident buffers.
     */
    inline void del() {
        if (deviceId < 0) return;
        if (elemsHost)
            free(elemsHost);
        if (elemsDevice) {
            CUDA_CHECK(cudaSetDevice(deviceId));
            CUDA_CHECK(cudaFree(elemsDevice));
        }
    }

    // /** Destructor **/
    // ~GRD() {
    //     del();
    // }
};

#endif  // GRD_H
