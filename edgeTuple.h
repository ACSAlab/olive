/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Yichao Cheng
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


/**
 * Edge Tuple.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2015-02-05
 * Last Modified: 2015-02-05
 */

#ifndef EDGE_TUPLE_H
#define EDGE_TUPLE_H


/**
 * COO representation. An edge is ternary tuple (`srcId`, `dstId`, `value`).
 * We use it be the immediate presentation.
 */
template<typename EdgeValue>
class EdgeTuple {
public:
    VertexId srcId;     /** The vertex id of the source vertex */
    VertexId dstId;     /** The vertex id of the target vertex */
    EdgeValue value;    /** The value associated with the edge */

    EdgeTuple(VertexId src, VertexId dst, EdgeValue v) {
        srcId = src;
        dstId = dst;
        value  = v;
    }
};

/**
 * Comparator for sorting COO edge tuple.
 */
template<typename EdgeTuple>
bool edgeTupleSrcCompare(EdgeTuple a, EdgeTuple b) {
    return a.srcId < b.srcId;
}

template<typename EdgeTuple>
bool edgeTupleDstCompare(EdgeTuple a, EdgeTuple b) {
    return a.dstId < b.dstId;
}

#endif // EDGE_TUPLE_H
