/**
 * GPU-resident Dataset.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-13
 * Last Modified: 2014-11-13
 */

#ifndef GRD_H
#define GRD_H

#include "logging.h"

/**
 * GPU-Resident Dataset (GRD) provides the utility for allocating data buffers
 * which can be transferred between CPU and GPU and accessed from both CPU and GPU.
 * GRD is allocated on host pinned memory, which is accessible by all CUDA contexts.
 */
template<typename T>
class GRD {
 private:
    T *     elemsHost;    /** Points to the host-allocated buffer */
    T *     elemsDevice;  /** Points to the GPU-allocated buffer */
    size_t  length;       /** The length of the buffer */
    int     deviceId;     /** The device GRD locates at */

 public:
    /** List Initializer */
    GRD(): elemsHost(NULL), elemsDevice(NULL), length(0), deviceId(-1) {}

    /**
     * Overloads the subscript to access an element on host side.
     * Do not check the boundaries for speed.
     */
    __host__ __device__
     inline T& operator[] (size_t index) const {
#ifdef __CUDA_ARCH__
        return elemsDevice[index];
#else
        return elemsHost[index];
#endif
    }

    /** Returning the size */
    inline size_t size() const {
        return length;
    }

    /**
     * Allocate the host- and device-resident buffer of length `len` on device
     * `id`.
     */
    inline void reserve(size_t len, int id) {
        assert(len > 0);
        assert(id >= 0);
        deviceId = id;
        length = len;
        elemsHost = reinterpret_cast<T *>(malloc(len * sizeof(T)));
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&elemsDevice),
                                  len * sizeof(T), cudaHostAllocPortable));
    }

    /** 
     * Write-backs the dataset from GPU's on-board memory to host memory.
     */
    inline void persist() {
        assert(length > 0);
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              length * sizeof(T), cudaMemcpyDefault));
    }

    /** 
     * Caches the dataset in GPU's on-board memory.
     */
    inline void cache() {
        assert(length > 0);
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              length * sizeof(T), cudaMemcpyDefault));
    }

    /** 
     * Free both host- and device- resident buffers.
     */
    inline void del() {
        if (elemsHost)
            free(elemsHost);
        if (elemsDevice) {
            CUDA_CHECK(cudaSetDevice(deviceId));
            CUDA_CHECK(cudaFreeHost(elemsDevice));
        }
    }

    /** Destructor **/
    ~GRD() {
        del();
    }
};

#endif  // GRD_H
