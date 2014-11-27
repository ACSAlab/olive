/**
 * Edge tuples
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-06
 * Last Modified: 2014-11-06
 */

#ifndef EDGE_TUPLE_H
#define EDGE_TUPLE_H

#include "common.h"

/**
 *  An edge is ternary tuple (`srcId`, `dstId`, `attr`)
 * 
 * @tparam ED type of the edge attribute
 */
template<typename ED>
class EdgeTuple {
 public:
    VertexId srcId;     /** The vertex id of the source vertex */
    VertexId dstId;     /** The vertex id of the target vertex */
    ED attr;            /** The attribute associated with the edge */

    /** Constructor with all three parameters */
    explicit EdgeTuple(VertexId src, VertexId dst, ED d) {
        srcId = src;
        dstId = dst;
        attr  = d;
    }
};

#endif  // EDGE_TUPLE_H
