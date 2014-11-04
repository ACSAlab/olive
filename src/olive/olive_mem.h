/**
 * An interface for CPU/GPU hybrid memory management
 *
 * The memory interface provide a universal view for memory management
 * TODO(onesuper): use templates to rewrite the interface
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-22
 * Last Modified: 2014-11-04
 */


#ifndef OLIVE_MEM_H
#define OLIVE_MEM_H

#include "olive_def.h"


/**
 * Memory operation type. We pass the type as parameter to 
 * differetiate the behavior for memory allocation.
 */
typedef enum {
    MEM_OP_HOST = 0,     // Allocated in the address space of CPU-side memory
    MEM_OP_DEVICE,       // Allocated in the address space of the GPU memory
    MEM_OP_HOST_PINNED,  // CPU-side pinned memory
    MEM_OP_HOST_MAPPED   // CPU-side pinned memory, but also mapped to the
                         // address space of the GPU memory. To refer the
                         // GPU-side space, cudaHostGetDevicePointer() must
                         // be called.
} MemOp;


/**
 * Allocate a memory buffer of size 'size' of memory type 'type'.
 * @param ptr pointer to allocated memory 
 * @param size buffer size to allocate
 * @param type type of the memory to allocate
 * @return SUCCESS if the buffer has been allocated, FAILURE otherwise
 */
Error oliveMalloc(void ** ptr, size_t size, MemOp memOp);


/**
 * Similar to olive_malloc() with the difference that the allocated space is
 * filled with zeros.
 * The buffer can be either pinned or not.
 * @param ptr pointer to allocated memory 
 * @param size buffer size to allocate
 * @param type type of the memory to allocate
 * @return SUCCESS if the buffer has been allocated, FAILURE otherwise
 */
Error oliveCalloc(void ** ptr, size_t size, MemOp memOp);


/**
 * Free a buffer allocated by olive_malloc() or olive_calloc() 
 * We rely on the user to pass in the correct 'type'
 * @param ptr pointer to the buffer to be freed
 * @param type type of the memory to allocate
 * @return SUCCESS if the buffer has been allocated, FAILURE otherwise
 */
Error oliveFree(void * ptr, MemOp memOp);

/**
 * Wraps the cudaMemcpy function (host->device).
 * With a platform having Uinified Virtual Space, cuda runtime can
 * detect the data transfer direction.
 * @param dest points to the device buffer
 * @param src points to the host buffer
 * @param size size of the buffer to copy
 */
#define oliveMemH2D(dest,  src,  size)  \
    ((cudaMemcpy(dest, src, size, cudaMemcpyDefault) == cudaSuccess) ? SUCCESS : FAILURE)

/**
 * Wraps the cudaMemcpy function (device->host).
 * @param dest points to the host buffer
 * @param src points to the device buffer
 * @param size size of the buffer to copy
 */
#define oliveMemD2H(dest, src, size)    \
    ((cudaMemcpy(dest, src, size, cudaMemcpyDefault) == cudaSuccess) ? SUCCESS : FAILURE)

/**
 * Wraps the cudaHostGetDevicePointer() function, which is used to fetch the
 * device pointer for the host-mapped buffer
 * @param deviceP points to the device buffer
 * @param hostP points to the host-mapped buffer
 */
#define oliveGetDeviceP(pDevice, pHost) \
    ((cudaHostGetDevicePointer(pDevice, pHost, 0) == cudaSuccess) ? SUCCESS : FAILURE)



#endif  // OLIVE_MEM_H
