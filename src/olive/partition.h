/**
 * Partition
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-13
 * Last Modified: 2014-11-13
 */


#ifndef PARTITION_H
#define PARTITION_H

#include "grd.h"

/**
 * Managing the resouce for each GPU-resident graph partition.
 * 

 *
 * Each graph parition is stored in (Compressed Sparsed Row) format for its
 * efficiency. In effect, CSR storage minimizes the memory footprint at the 
 * expense of bringing indirect memory access.
 */
class Partition {
 public:
    /**
     * @note It is always a convention that each partion is assigned to a device
     * with the same logical ID. e.g., `Partiton 0` is assigned to `Device 0`.
     */
    PartitionId     partitionId;

    /**
     * Stores the starting indices of vertices' outgoing edges.
     * For example, vertices[i] tells where its outgoing edges locate. 
     * The number of it outgoing edges is given by vertices[i+1] - vertices[i].
     */
    GRD<EdgeId>     vertices;

    /**
     * Stores the destination vertex id to represent a complete outgoing edge. 
     */
    GRD<VertexId>   edges;

    ghostVertices;

    RoutingTable    outBoxTable;
    RoutingTable    inBoxTable;

    /**
     * Enables overlapped communication and computation.
     * e.g., the first stream is used to launch communication operations,
     * while the second one is used to launch computational kernels.
     */
    cudaStream_t    streams[2];

    /**
     * Measures the execution time of a certain kernel.
     */
    cudaEvent_t     startEvent;
    cudaEvent_t     endEvent;

 public:

    Partition(PartitionId pid) {
        partitionId = pid;
        CALL_SAFE(cudaSetDevice(partitionId));
        CALL_SAFE(cudaStreamCreate(&streams[0]));
        CALL_SAFE(cudaStreamCreate(&streams[1]));
        CALL_SAFE(cudaEventCreate(&startEvent));
        CALL_SAFE(cudaEventCreate(&endEvent));
    }


    ~Partion() {
        CALL_SAFE(cudaStreamDestroy(streams[0]));
        CALL_SAFE(cudaStreamDestroy(streams[1]));
        CALL_SAFE(cudaEventDestroy(startEvent));
        CALL_SAFE(cudaEventDestroy(endEvent));
    }

}



#endif  // PARTITION_H 