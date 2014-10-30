/**
 * An interface for CPU/GPU hybrid memory management
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-22
 * Last Modified: 2014-10-23
 */


#include "olive_def.h"
#include "olive_mem.h"


Error oliveMalloc(void ** ptr, size_t size, MemOperation type) {
    switch (type) {
    case OLIVE_MEM_HOST:
        * ptr = malloc(size);
        if (* ptr == NULL) return FAILURE;
        break;
    case OLIVE_MEM_HOST_PINNED:
        if (cudaMallocHost(ptr, size, cudaHostAllocPortable) != cudaSuccess)
            return FAILURE;
        break;
    case OLIVE_MEM_HOST_MAPPED:
        /**
         * cudaHostAllocMapped maps the allocation into the CUDA address space.
         * The device pointer to the memory may be obtained by calling
         * cudaHostGetDevicePointer().
         *
         * cudaHostAllocWriteCombined memory can be transferred across the PCI
         * Express bus more quickly on some system configurations, but cannot
         * be read efficiently by most CPUs. So it is a good option for 
         * host->device transfers.
         */
        if (cudaMallocHost(ptr, size, cudaHostAllocPortable |
                                      cudaHostAllocMapped |
                                      cudaHostAllocWriteCombined)
            != cudaSuccess) return FAILURE;
        break;
    case OLIVE_MEM_DEVICE:
        if (cudaMalloc(ptr, size) != cudaSuccess) return FAILURE;
        break;
    default:
        oliveFatal("invalid memory type");
    }
    return SUCCESS;
}

Error oliveCalloc(void ** ptr, size_t size, MemOperation type) {
    if (oliveMalloc(ptr, size, type) != SUCCESS) return FAILURE;
    switch (type) {
    case OLIVE_MEM_HOST:
    case OLIVE_MEM_HOST_PINNED:
    case OLIVE_MEM_HOST_MAPPED:
        memset(* ptr, 0, size);  // always succeed, guaranteed by OS
        break;
    case OLIVE_MEM_DEVICE:
        if (cudaMemset(* ptr, 0, size) != cudaSuccess) return FAILURE;
        break;
    default:
        oliveFatal("invalid memory type");
    }
    return SUCCESS;
}

Error oliveFree(void * ptr, MemOperation type) {
    switch (type) {
    case OLIVE_MEM_HOST:
        free(ptr);   // always succeed, guaranteed by OS
        break;
    case OLIVE_MEM_HOST_PINNED:
    case OLIVE_MEM_HOST_MAPPED:
        if (cudaFreeHost(ptr) != cudaSuccess) return FAILURE;
        break;
    case OLIVE_MEM_DEVICE:
        if (cudaFree(ptr) != cudaSuccess) return FAILURE;
        break;
    default:
        oliveFatal("invalid memory type");
    }
    return SUCCESS;
}

