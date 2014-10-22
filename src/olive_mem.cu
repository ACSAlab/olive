/**
 * An interface for CPU/GPU hybrid memory management
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-22
 * Last Modified: 2014-10-22
 *
 */

#include "olive_def.h"
#include "olive_mem.h"


error_t olive_malloc(size_t size, olive_mem_t type, void ** ptr) {
    error_t err = SUCCESS;
    switch (type) {
    case OLIVE_MEM_HOST:
        *ptr = malloc(size);
        if (*ptr == NULL) {
            err = FAILURE;
        }
        break;
    case OLIVE_MEM_HOST_PINNED:
        if (cudaMallocHost(ptr, size, cudaHostAllocPortable) != cudaSuccess) {
            err = FAILURE;
        }
        break;
    case OLIVE_MEM_HOST_MAPPED:
        if (cudaMallocHost(ptr, size, cudaHostAllocPortable |
                           cudaHostAllocMapped | cudaHostAllocWriteCombined)
            != cudaSuccess) {
            err = FAILURE;
        }
        break;
    case OLIVE_MEM_DEVICE:
        if (cudaMalloc(ptr, size) != cudaSuccess) {
            err = FAILURE;
            // Here we assume the failure is caused by insuffient memory
            // and print the memory size out
            size_t available = 0; size_t total = 0;             
            CUT_SAFE_CALL(cudaMemGetInfo(&available, &total));
            print_error("insufficient device memory (%llu is requested, %llu is available)",
                        size, available);
        }
        break;
    default:
        print_error("invalid memory type");
        assert(0);  
    }
    return err;
 }

error_t olive_calloc(voi d** ptr, size_t size, totem_mem_t type) {
    if (olive_malloc(ptr, size, type) != SUCCESS) {
        return FAILURE;
    }
    error_t err = SUCCESS;
    switch (type) {
    case TOTEM_MEM_HOST:
    case TOTEM_MEM_HOST_PINNED:
    case TOTEM_MEM_HOST_MAPPED:
        memset(*ptr, 0, size);
        break;
    case TOTEM_MEM_DEVICE:
        if (cudaMemset(*ptr, 0, size) != cudaSuccess) {
            err = FAILURE;
        }
        break;
    default:
        print_error("invalid memory type");
        assert(0);
    }
    return err;
}

void olive_free(void * ptr, olive_mem_t type) {
    switch (type) {
    case OLIVE_MEM_HOST:
        free(ptr);
        break;
    case OLIVE_MEM_HOST_PINNED:
    case OLIVE_MEM_HOST_MAPPED:
        CUT_SAFE_CALL(cudaFreeHost(ptr));
        break;
    case OLIVE_MEM_DEVICE:
        CUT_SAFE_CALL(cudaFree(ptr));
        break;
    default:
        print_error("invalid memory type");
        assert(0);  
    }
}

 
