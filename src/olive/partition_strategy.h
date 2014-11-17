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

#include "common.h"
#include "utils.h"


/** 
 * An abstract interface for partitioning vertices.
 */
class PartitionStrategy {
 public:
    /** 
     * Returns the partition number for a given vertex.
     * @param  vid       Source vertex of the edge
     * @param  numParts  Number of parts to partition
     * @return           The partition number for a given vertex
     */
    virtual PartitionId getPartition(VertexId vid, PartitionId numParts) = 0;
};

class RandomEdgeCut: public PartitionStrategy {
    PartitionId getPartition(VertexId vid, PartitionId numParts) {
        return util::hashCode(vid) % numParts;
    }
};


#endif  // PARTION_STRATEGY_H
