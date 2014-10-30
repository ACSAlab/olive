/**
 * An interface for CPU/GPU hybrid memory management
 *
 * The memory interface provide a universal view for memory management
 * TODO(onesuper): use templates to rewrite the interface
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-22
 * Last Modified: 2014-10-22
 */


#ifndef OLIVE_MEM_H
#define OLIVE_MEM_H


/**
 * Memory operation type. We pass the type as parameter to 
 * differetiate the behavior (allocate or free).
 */
typedef enum {
    OLIVE_MEM_HOST = 0,     // Allocated in the address space of CPU-side memory
    OLIVE_MEM_DEVICE,       // Allocated in the address space of the GPU memory
    OLIVE_MEM_HOST_PINNED,  // CPU-side pinned memory
    OLIVE_MEM_HOST_MAPPED   // CPU-side pinned memory, but also mapped to the
                            // address space of the GPU memory. To refer the
                            // GPU-side space, cudaHostGetDevicePointer() must
                            // be called.
} MemOperation;


/**
 * Allocate a memory buffer of size 'size' of memory type 'type'.
 * @param[out] ptr: a pointer to the allocated space 
 * @param[in] size: buffer size to allocate
 * @param[in] type: type of the memory to allocate
 * @return SUCCESS if the buffer has been allocated, FAILURE otherwise
 */
Error oliveMalloc(void ** ptr, size_t size, MemOperation type);


/**
 * Similar to olive_malloc() with the difference that the allocated space is
 * filled with zeros.
 * The buffer can be either pinned or not.
 * @param[out] ptr: a pointer to the allocated space 
 * @param[in] size: buffer size to allocate
 * @param[in] type: type of the memory to allocate
 * @return SUCCESS if the buffer has been allocated, FAILURE otherwise
 */
Error oliveCalloc(void ** ptr, size_t size, MemOperation type);


/**
 * Free a buffer allocated by olive_malloc() or olive_calloc() 
 * We rely on the user to pass in the correct 'type'
 * @param[in] ptr: pointer to the buffer to be freed
 * @param[in] type: type of the memory to allocate
 * @return SUCCESS if the buffer has been allocated, FAILURE otherwise
 */
Error oliveFree(void * ptr, MemOperation type);


#endif  // OLIVE_MEM_H
