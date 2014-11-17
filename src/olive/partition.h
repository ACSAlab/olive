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
 * Manages the resouce of each GPU-resident graph partition
 */
class Partition {

    PartitionId     partitionId;
    PartitionId     processorId;

    GRD<EdgeId>     vertices;
    GRD<VertexId>   edges;

    RoutingTable    outBoxTable;
    RoutingTable    inBoxTable;

    // Enables overlapped communication and computation.
    // e.g., the first stream is used to launch communication operations,
    // while the second one is used to launch computational kernels.
    cudaStream_t    streams[2];

    // Measures the execution time of a certain kernel
    cudaEvent_t     startEvent;
    cudaEvent_t     endEvent;

 public:
    Partition(PartitionId pid) {
        processorId = pid;
        CALL_SAFE(cudaSetDevice(processorId));
        CALL_SAFE(cudaStreamCreate(&streams[0]));
        CALL_SAFE(cudaStreamCreate(&streams[1]));
        CALL_SAFE(cudaEventCreate(&startEvent));
        CALL_SAFE(cudaEventCreate(&endEvent));
    }


    ~Partion() {
        CALL_SAFE(cudaSetDevice(processorId));
        CALL_SAFE(cudaStreamDestroy(streams[0]));
        CALL_SAFE(cudaStreamDestroy(streams[1]));
        CALL_SAFE(cudaEventDestroy(startEvent));
        CALL_SAFE(cudaEventDestroy(endEvent));
    }

}



#endif  // PARTITION_H 