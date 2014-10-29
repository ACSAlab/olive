/**
 * An interface for CPU/GPU hybrid memory management
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-22
 * Last Modified: 2014-10-22
 */


#ifndef OLIVE_MEM_H
#define OLIVE_MEM_H

#include "olive_def.h"

/**
 * Memory operation type. We pass the type as parameter to 
 * differetiate the behavior (allocate or free).
 */
typedef enum {
    OLIVE_MEM_HOST = 0,     // Allocated in the address space of CPU-side memory 
    OLIVE_MEM_HOST_PINNED,  // CPU-side pinned memory 
    OLIVE_MEM_HOST_MAPPED,  // CPU-side pinned memory, but also mapped to the 
                            // address space of the GPU memory. To refer the GPU-side
                            // space, cudaHostGetDevicePointer() must be called.
    OLIVE_MEM_DEVICE        // Allocated in the address space of the GPU's memory
} olive_mem_t;


/**
 * Allocate a memory buffer of size 'size' of memory type 'type'.
 * @param[out] ptr: a pointer to the allocated space 
 * @param[in] size: buffer size to allocate
 * @param[in] type: type of the memory to allocate
 * @return SUCCESS if the buffer has been allocated, FAILURE otherwise
 */
error_t olive_malloc(void ** ptr, size_t size, olive_mem_t type);


/**
 * Similar to olive_malloc() with the difference that the allocated space is
 * filled with zeros.
 * the buffer can be either pinned or not.
 * @param[out] ptr: a pointer to the allocated space 
 * @param[in] size: buffer size to allocate
 * @param[in] type: type of the memory to allocate
 * @return SUCCESS if the buffer has been allocated, FAILURE otherwise
 */
error_t totem_calloc(void ** ptr, size_t size, olive_mem_t type);


/**
 * Free a buffer allocated by olive_malloc() or olive_calloc() 
 * We rely on the user to pass in the correct 'type'
 * @param[in] ptr: pointer to the buffer to be freed
 * @param[in] type: type of the memory to allocate
 */
void olive_free(void * ptr, olive_mem_t type);


#endif // OLIVE_MEM_H