/**
 * Strategies for partitioning graph
 *
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-04
 * Last Modified: 2014-11-04
 */

#ifndef PARTITION_STRATEGY_H
#define PARTITION_STRATEGY_H

#include "Defines.h"
#include "util/Utils.h"
/** 
 * An abstract interface for partitioning edges which is based on the 
 * vertexId of each vertex.
 */
class PartitionStrategy {
 public:
    /** 
     * Returns the partition number for a given edge.
     * @param  srcId     Source vertex of the edge
     * @param  dstId     Destination vertex of the edge
     * @param  numParts  Number of parts to partition
     * @return           The partition number for a given vertex
     */
    virtual PartitionId getPartition(VertexId srcId, VertexId dstId, PartitionId numParts) = 0;
};


class RandomVertexCut: public PartitionStrategy {
    PartitionId getPartition(VertexId srcId, VertexId dstId, PartitionId numParts) {
        return util::hashCode(srcId + dstId) % numParts;
    }
};




#endif  // PARTION_STRATEGY_H
