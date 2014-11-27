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
    explicit GRD() : elemsHost(NULL), elemsDevice(NULL), length(0), deviceId(0) {}

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
    inline size_t size(void) const {
        return length;
    }

    /**
     * Allocate the host- and device-resident buffer of length `len` on device
     * `id`.
     */
    inline void reserve(size_t len, int id) {
        deviceId = id;
        length = len;
        assert(len > 0);
        elemsHost = reinterpret_cast<T *>(malloc(len * sizeof(T)));
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&elemsDevice),
                                  len * sizeof(T), cudaHostAllocPortable));
        LOG(DEBUG) << length << " data reserve on GPU" << deviceId;
    }

    /** 
     * Write-backs the dataset from GPU's on-board memory to host memory.
     */
    inline void persist(void) {
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              length * sizeof(T), cudaMemcpyDefault));
        LOG(DEBUG) << length << " data persist on GPU " << deviceId;
    }

    /** 
     * Caches the dataset in GPU's on-board memory.
     */
    inline void cache(void) {
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMemcpy(elemsDevice, elemsHost,
                              length * sizeof(T), cudaMemcpyDefault));
        LOG(DEBUG) << length << " data cache on GPU " << deviceId;
    }

    /** 
     * Free both host- and device- resident buffers.
     */
    inline void del(void) {
        if (elemsHost)   free(elemsHost);
        CUDA_CHECK(cudaSetDevice(deviceId));
        if (elemsDevice) {
            CUDA_CHECK(cudaSetDevice(deviceId));
            CUDA_CHECK(cudaFreeHost(elemsDevice));
            LOG(DEBUG) << length << " data delete on GPU " << deviceId;
        }
    }

    ~GRD(void) {
        del();
    }
};

#endif  // GRD_H
