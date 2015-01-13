/**
 * Edge tuples.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-06
 * Last Modified: 2014-11-06
 */

#ifndef EDGE_TUPLE_H
#define EDGE_TUPLE_H

#include "common.h"

/**
 *  An edge is ternary tuple (`srcId`, `dstId`, `value`).
 *
 * @tparam EdgeValue type of the edge value
 */
template<typename EdgeValue>
class EdgeTuple {
public:
    VertexId srcId;     /** The vertex id of the source vertex */
    VertexId dstId;     /** The vertex id of the target vertex */
    EdgeValue value;     /** The value associated with the edge */

    /** Constructor with all three parameters */
    explicit EdgeTuple(VertexId src, VertexId dst, EdgeValue v) {
        srcId = src;
        dstId = dst;
        value  = v;
    }
};

#endif  // EDGE_TUPLE_H
