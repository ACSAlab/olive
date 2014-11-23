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
 public:
    T *     elemsHost;    // Points to the host-allocated buffer
    T *     elemsDevice;  // Points to the GPU-allocated buffer
    size_t  length;       // The length of the buffer

    /**
     * Allocate the host- and device-resident buffer of length `len`.
     */
    explicit GRD(size_t len) {
        length = len;
        elemsHost = reinterrupt_cast<T *> malloc(elemsHost, len * sizeof(T));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&elemsDevice),
                                  len * sizeof(T), cudaHostAllocPortable));
    }

    /**
     * Overloads the subscript to access an element on host side.
     * Do not check the boundaries for speed.
     */
    __host__
    inline T& operator[] (int index) const {
        return elemsHost[index];
    }

    /**
     * Overloads the subscript to access an element on device side.
     */
    __device__
    inline T& operator[] (int index) const {
        return elemsDevice[index];
    }

    /** Write-back the dataset from GPU's onboard memory to host memory */
    void persist(void) {
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              size * sizeof(T), cudaMemcpyDefault));
    }

    /** Cache the dataset in GPU's onboard memory */
    void cache(void) {
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              size * sizeof(T), cudaMemcpyDefault));
    }

    /** Free both host- and device- resident buffers */
    ~GRD(void) {
        free(elemsHost);
        CUDA_CHECK(cudaFreeHost(elemsHost));
    }
};

#endif  // GRD_H