/**
 * An interface for CPU/GPU hybrid memory management
 *
 * The memory interface provide a universal view for memory management
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-22
 * Last Modified: 2014-11-04
 */

#ifndef MEMORY_H
#define MEMORY_H

#include "common.h"
#include "Defines.h"
#include "Logging.h"


/**
 * Wraps around the cuda memory API and provides a unified interface on
 * different `MemoryLevel`.
 */
class Memory {
 public:
    /**
     * Allocate a memory buffer at a specified `memoryLevel`.
     * 
     * @param ptr         A Pointer to allocated memory 
     * @param size        Buffer size to allocate
     * @param memoryLevel Level of memory to allocate at
     * @return            SUCCESS if the buffer has been allocated
     */
    static Error malloc(void ** ptr, size_t size,
                        MemoryLevel memoryLevel = CPU_ONLY)
    {
        cudaError_t err;
        switch (memoryLevel) {
        case CPU_ONLY:
            *ptr = ::malloc(size);
            if (*ptr == NULL) {
                Logging::logError("Malloc fail");
                return FAILURE;
            }
            break;

        case MANAGED:
            err = cudaMallocManaged(ptr, size);
            if (err != cudaSuccess) {
				Logging::logError("CUDA: %s", cudaGetErrorString(err));
                return FAILURE;
            }
            break;

        case GPU_ONLY:
            err = cudaMalloc(ptr, size);
            if (err != cudaSuccess) {
				Logging::logError("CUDA: %s", cudaGetErrorString(err));
                return FAILURE;
            }
            break;

        default:
			Logging::logError("invalid memory operaion type");
            return FAILURE;
        }
        return SUCCESS;
    }

    /**
     * Allocate a memory buffer filled with a specified `value`.
     * 
     * @param ptr         Pointer to allocated memory
     * @param value       Specified value
     * @param size        Buffer size to allocate
     * @param memoryLevel Level of memory to allocate at
     * @return            SUCCESS if the buffer has been set
     */
    static Error memset(void ** ptr, int value, size_t size,
                        MemoryLevel memoryLevel = CPU_ONLY)
    {
        cudaError_t err;
        switch (memoryLevel) {
        case CPU_ONLY:
            ::memset(*ptr, value, size);
            break;

        case MANAGED:
        case GPU_ONLY:
            err = cudaMemset(* ptr, value, size);
            if ((err != cudaSuccess)) {
				Logging::logError("CUDA: %s", cudaGetErrorString(err));
                return FAILURE;
            }
            break;

        default:
			Logging::logError("Invalid memory operaion type");
            return FAILURE;
        }
        return SUCCESS;
    }

    /**
     * Free a specified buffer.
     * 
     * @param ptr         Pointer to the buffer to be freed
     * @param memoryLevel Level of the memory to allocate at
     * @return            SUCCESS if the buffer has been allocated
     */
    static Error free(void * ptr, MemoryLevel memoryLevel = CPU_ONLY) {
        cudaError_t err;
        switch (memoryLevel) {
        case CPU_ONLY:
            free(ptr);
            break;

        case MANAGED:
        case GPU_ONLY:
            err = cudaFree(ptr);
            if (err != cudaSuccess) {
				Logging::logError("CUDA: %s", cudaGetErrorString(err));
                return FAILURE;
            }
            break;

        default:
			Logging::logError("Invalid memory operaion type");
            return FAILURE;
        }
        return SUCCESS;
    }

    /**
     * Wraps the cudaMemcpy() function using Peer-to-Peer copy.
     * 
     * @note Must be carried out on a platform supporting unified memory.
     * 
     * @param  dst   Points to the device buffer
     * @param  src   Points to the host buffer
     * @param  size  Size of the buffer to copy
     * @return       SUCCESS if the buffer has been copyed, FAILURE otherwise
     */
    static inline Error memcpy(void * dst, const void * src, size_t size) {
        if (cudaMemcpy(dst, src, size, cudaMemcpyDefault)== cudaSuccess)
            return SUCCESS;
        else
            return FAILURE;
    }
};

#endif  // MEMORY_H
