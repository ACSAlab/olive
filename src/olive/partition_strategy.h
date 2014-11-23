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
 * An abstract interface for partitioning vertices for flexible graph 
 * representation. 
 * 
 * TODO(onesuper): Paritioning according to the topology of graph.
 * A partition strategy may possibly consider some of the graph attributes,
 * e.g. outdegree, etc.
 */
class PartitionStrategy {
 public:
    /** 
     * Returns the partition number for a given vertex.
     * @param  id        Id for the vertex in graph
     * @param  numParts  Number of parts to partition
     * @return           The partition number for a given vertex
     */
    virtual PartitionId getPartition(VertexId id, PartitionId numParts) const = 0;
};

class RandomEdgeCut: public PartitionStrategy {
    PartitionId getPartition(VertexId id, PartitionId numParts) const {
        return util::hashCode(id) % numParts;
    }
};


#endif  // PARTION_STRATEGY_H
