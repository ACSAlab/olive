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
 * The CUDA kernel of the engine.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-12-20
 * Last Modified: 2014-12-20
 */

#ifndef OLIVER_KERNEL_H
#define OLIVER_KERNEL_H

#include "common.h"




/**
 * The CUDA kernel for expanding vertices in the work queue.
 */
template<typename VertexValue,
         typename AccumValue,
         typename EdgeValue,
         typename F>
__global__
void edgeFilterDenseKernel(
    const VertexId *workqueue,
    const VertexId *workqueueSize,
    const EdgeId   *vertices,
    const VertexId *outgoingEdges,
    VertexValue    *vertexValues,
    AccumValue     *accumulators,
    EdgeValue      *edgeValues,
    int            *workset,
    F f)
{
    VertexId pos = THREAD_INDEX;
    if (pos >= *workqueueSize) return;
    VertexId srcId = workqueue[pos];

    EdgeId start = vertices[srcId];
    EdgeId end = vertices[srcId + 1];
    EdgeId outdegree = end - start;
    VertexValue srcValue = vertexValues[srcId];

    for (EdgeId e = start; e < end; e ++) {
        // Edge level parallelism, which is exploited by SIMD lanes
        AccumValue accum = f.gather(srcValue, outdegree, edgeValues[e]);
        VertexId dstId = outgoingEdges[e];
        f.reduce(accumulators[dstId], accum);
        workset[dstId] = 1;
    }
}

template<typename VertexValue,
         typename AccumValue,
         typename EdgeValue,
         typename F>
__global__
void edgeFilterSparseKernel(
    const int      *worksetSrc,
    VertexId       worksetsize,
    const EdgeId   *vertices,
    const VertexId *outgoingEdges,
    VertexValue    *vertexValues,
    AccumValue     *accumulators,
    EdgeValue      *edgeValues,
    int            *worksetDst,
    F f)
{
    VertexId srcId = THREAD_INDEX;
    if (srcId >= worksetsize) return;
    if (!worksetSrc[srcId]) return;

    EdgeId start = vertices[srcId];
    EdgeId end = vertices[srcId + 1];
    EdgeId outdegree = end - start;
    VertexValue srcValue = vertexValues[srcId];

    for (EdgeId e = start; e < end; e ++) {
        // Edge level parallelism, which is exploited by SIMD lanes
        AccumValue accum = f.gather(srcValue, outdegree, edgeValues[e]);
        VertexId dstId = outgoingEdges[e];
        f.reduce(accumulators[dstId], accum);
        worksetDst[dstId] = 1;
    }
}


/**
 * The vertex map kernel.
 */
template<typename VertexValue,
         typename AccumValue,
         typename EdgeValue,
         typename F>
__global__
void edgeMapKernel(
    const int      *workset,
    VertexId       worksetsize,
    const EdgeId   *vertices,
    const VertexId *outgoingEdges,
    VertexValue    *vertexValues,
    AccumValue     *accumulators,
    EdgeValue      *edgeValues,
    F f)
{
    VertexId srcId = THREAD_INDEX;
    if (srcId >= worksetsize) return;
    if (!workset[srcId]) return;

    EdgeId start = vertices[srcId];
    EdgeId end = vertices[srcId + 1];
    EdgeId outdegree = end - start;
    VertexValue srcValue = vertexValues[srcId];

    for (EdgeId e = start; e < end; e ++) {
        // Edge level parallelism, which is exploited by SIMD lanes
        AccumValue accum = f.gather(srcValue, outdegree, edgeValues[e]);
        VertexId dstId = outgoingEdges[e];
        f.reduce(accumulators[dstId], accum);
    }
}



template<typename VertexValue,
         typename AccumValue,
         typename F>
__global__
void vertexFilterDenseKernel(
    const int   *workset,
    VertexId     worksetsize,
    VertexValue *vertexValues,
    AccumValue  *accumulators,
    VertexId    *workqueue,
    VertexId    *workqueueSize,
    F f)
{
    int v = THREAD_INDEX;
    if (v >= worksetsize) return;
    if (!workset[v]) return;

    if (f.cond(vertexValues[v], v)) {
        f.update(vertexValues[v], accumulators[v]);
        VertexId pos = atomicAdd(workqueueSize, 1);
        workqueue[pos] = v;
    }
}

template<typename VertexValue,
         typename AccumValue,
         typename F>
__global__
void vertexFilterSparseKernel(
    const int   *worksetSrc,
    VertexId     worksetsize,
    VertexValue *vertexValues,
    AccumValue  *accumulators,
    int         *worksetDst,
    F f)
{
    int v = THREAD_INDEX;
    if (v >= worksetsize) return;
    if (!worksetSrc[v]) return;

    if (f.cond(vertexValues[v], v)) {
        f.update(vertexValues[v], accumulators[v]);
        worksetDst[v] = 1;
    }
}


/**
 * The vertex map kernel.
 */
template<typename VertexValue,
         typename AccumValue,
         typename F>
__global__
void vertexMapSparseKernel(
    const int    *workset,
    VertexId      worksetsize,
    VertexValue  *vertexValues,
    AccumValue   *accumulators,
    F f)
{
    int v = THREAD_INDEX;
    if (v >= worksetsize) return;
    if (!workset[v]) return;

    f(vertexValues[v], accumulators[v]);
}


/**
 * The vertex map kernel.
 */
template<typename VertexValue,
         typename AccumValue,
         typename F>
__global__
void vertexMapDenseKernel(
    const VertexId  *workqueue,
    const VertexId  *workqueueSize,
    VertexValue     *vertexValues,
    AccumValue      *accumulators,
    F f)
{
    VertexId pos = THREAD_INDEX;
    if (pos >= *workqueueSize) return;
    VertexId v = workqueue[pos];
    f(vertexValues[v], accumulators[v]);
}

#endif  // OLIVER_KERNEL_H
