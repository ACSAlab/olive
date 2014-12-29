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

    /** Returning the size */
    inline size_t size() const {
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
     * Set all the elements to the value `x` on the host, then caches.
     */
    void allTo(T x) {
        for (size_t i = 0; i < length; i++) {
            elemsHost[i] = x;
        }
        cache();
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


    inline void print() {
        persist();
        for (int i = 0; i < length; i++) {
            printf("%d ", elemsHost[i]);
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

    /** Destructor **/
    ~GRD() {
        del();
    }
};

#endif  // GRD_H
