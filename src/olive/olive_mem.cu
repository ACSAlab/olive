/**
 * Implementation of the interface for CPU/GPU hybrid memory management
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-22
 * Last Modified: 2014-11-04
 */


#include "olive_mem.h"


Error oliveMalloc(void ** ptr, size_t size, MemOp memOp) {
    cudaError_t err;
    switch (memOp) {
    case MEM_OP_HOST:
        * ptr = malloc(size);
        if (* ptr == NULL) {
            oliveError("malloc fail");
            return FAILURE;
        }
        break;
    case MEM_OP_HOST_PINNED:
        err = cudaMallocHost(ptr, size, cudaHostAllocPortable);
        if (err != cudaSuccess) {
            oliveCudaError(err);
            return FAILURE;
        }
        break;
    // Maps the allocation into the CUDA address space.
    // The device pointer to the memory may be obtained by calling
    // cudaHostGetDevicePointer().
    case MEM_OP_HOST_MAPPED:
        err = cudaMallocHost(ptr, size, cudaHostAllocPortable |
                                        cudaHostAllocMapped |
                                        cudaHostAllocWriteCombined);
        // 'cudaHostAllocWriteCombined' memory can be transferred across the PCI
        // Express bus more quickly on some system configurations, but cannot
        // be read efficiently by most CPUs. So it is a good option for
        // host->device transfers.
        if (err != cudaSuccess) {
            oliveCudaError(err);
            return FAILURE;
        }
        break;
    case MEM_OP_DEVICE:
        if ((err = cudaMalloc(ptr, size)) != cudaSuccess) {
            oliveCudaError(err);
            return FAILURE;
        }
        break;
    default:
        oliveError("invalid memory operaion type");
        return FAILURE;
    }
    return SUCCESS;
}

Error oliveCalloc(void ** ptr, size_t size, MemOp memOp) {
    if (oliveMalloc(ptr, size, memOp) != SUCCESS) return FAILURE;
    cudaError_t err;
    switch (memOp) {
    case MEM_OP_HOST:
    case MEM_OP_HOST_PINNED:
    case MEM_OP_HOST_MAPPED:
        memset(* ptr, 0, size);
        break;
    case MEM_OP_DEVICE:
        if ((err = cudaMemset(* ptr, 0, size)) != cudaSuccess) {
            oliveCudaError(err);
            return FAILURE;
        }
        break;
    default:
        oliveError("invalid memory operaion type");
        return FAILURE;
    }
    return SUCCESS;
}

Error oliveFree(void * ptr, MemOp memOp) {
    cudaError_t err;
    switch (memOp) {
    case MEM_OP_HOST:
        free(ptr);
        break;
    case MEM_OP_HOST_PINNED:
    case MEM_OP_HOST_MAPPED:
        if ((err = cudaFreeHost(ptr)) != cudaSuccess) {
            oliveCudaError(err);
            return FAILURE;
        }
        break;
    case MEM_OP_DEVICE:
        if ((err = cudaFree(ptr)) != cudaSuccess) {
            oliveCudaError(err);
            return FAILURE;
        }
        break;
    default:
        oliveError("invalid memory operaion type");
        return FAILURE;
    }
    return SUCCESS;
}

