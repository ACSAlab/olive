/**
 * GPU-resident Dataset
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-13
 * Last Modified: 2014-11-13
 */

#ifndef GRD_H
#define GRD_H

#include "logging.h"

/**
 * GPU-Resident Dataset (GRD) is allocated on host pinned memory, which is
 * accessible by all CUDA contexts.
 */
template<typename T>
class GRD {
    size_t size;
    T * elemsHost;
    T * elemsDevice;

    explicit GRD(size_t size_) {
        size = size_;
        elemsHost = reinterrupt_cast<T *> malloc(elemsHost, size);
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&elemsDevice),
                                  size * sizeof(T), cudaHostAllocPortable));
    }

    /**
     * Overloads the subscript to access an element on host side.
     * Do not check the boundaries for speed.
     */
    __host__
    inline T & operator[] (int index) const {
        return elemsHost[index];
    }

    /**
     * Overloads the subscript to access an element on device side.
     */
    __device__
    inline T & operator[] (int index) const {
        return elemsDevice[index];
    }

    /** Write the dataset from GPU's onboard memory to host memory */
    void persist(void) {
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              size * sizeof(T), cudaMemcpyDefault));
    }

    /** Cache the dataset in GPU's onboard memory */
    void cache(void) {
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              size * sizeof(T), cudaMemcpyDefault));
    }

    ~GRD(void) {
        free(elemsHost);
        CUDA_CHECK(cudaFreeHost(elemsHost));
    }
};

#endif  // GRD_H
