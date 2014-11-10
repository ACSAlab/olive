/**
 * 
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-06
 * Last Modified: 2014-11-06
 */


#ifndef PARTITION_BUILDER_H
#define PARTITION_BUILDER_H

#include <vector>
#include "Define.h"
#include "EdgeTuple.h"
#include "PartitionStrategy.h"

/**
 * Contructs edge partitions from an edge tuples array.
 * 
 * @tparam ED type of the edge attribute
 */
template<ED>
class PartitionBuilder {
 private:
    std::vector<Edge> * edgeTuples;

 public:
    PartitionBuilder(void) {
        edgeTuples = new std::vector<EdgeTuple>;
    }

    /**
     * Add an edge tuple to the array
     * @param src The vertex id of the source vertex
     * @param dst The vertex id of the target vertex
     * @param d   The attribute associated with the edge
     */
    void add(VertexId src, VertexId dst, ED d) {
        edgeTuples.add(EdgeTuple(src, dst, d));
    }

    /** 
     * Create an vertex partition from the edge set with a specified 
     * `PartionStrategy`.
     * @return  [description]
     */
    Partition & Partition(void) {

    }

    ~PartitionBuilder() {
        delete edgeTuples;
    }
};


#endif  // PARTITION_BUILDER_H

