/**
 * Defines the interface for the device-resilient graph data structure
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-02
 * Last Modified: 2014-11-04
 */


#ifndef OLIVE_GRAPHD_H
#define OLIVE_GRAPHD_H

#include "olive_def.h"
#include "olive_graphH.h"

/**
 * Memory kind of GraphD 
 */
typedef enum {
    GRAPHD_MEM_DEVICE = 0,  // Allocated in the address space of the GPU memory
    GRAPHD_MEM_HOST_MAPPED  // CPU-side pinned memory, but also mapped to the
                            // address space of the GPU memory. To refer the
                            // GPU-side space, cudaHostGetDevicePointer() must
                            // be called.
} GraphDMem;

/**
 * GraphD defines the GPU-resilient graph.
 */
class GraphD: public Graph {
 public:
    /**
     * Indicates the memory kind
     */
    GraphDMem dmem;

    /**
     * Stores the host mapped vertex list.
     * Valid only when the GraphD is allocated in the host-mapped memory.
     * Otherwise it is a NULL pointer.
     */
    EdgeId   * vertexListMapped;

    /**
     * Stores the host mapped edge list.
     * Valid only when the GraphD is allocated in the host-∏mapped memory.
     * Otherwise it is a NULL pointer.
     */   
    VertexId * edgeListMapped;

    /**
     * Constructor
     */
    GraphD(void): Graph(), vertexListMapped(NULL), edgeListMapped(NULL) {}

    /**
     * Builds GraphD object from a host graph object. 
     * By default, it is allocated on device (GRAPHD_MEM_DEVICE).
     * 
     * @param graph refers to a graph built on the host side
     * @param mem indicates where GraphD buffers will locate. 
     * @return SUCCESS if built, FAILURE otherwise 
     */
    Error fromGraphH(const GraphH & graphH, GraphDMem mem = GRAPHD_MEM_DEVICE);

    /**
     * Cleans up all the allocated buffers
     */
    void finalize(void);

    /**
     * Deconstructor
     */
    ~GraphD(void) { finalize(); }
};

#endif  // OLIVE_GRAPHD
