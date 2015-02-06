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
         typename F>
__global__
void edgeMapKernel(
    const VertexId *workqueue,
    const VertexId *workqueueSize,
    const EdgeId   *vertices,
    const VertexId *outgoingEdges,
    VertexValue    *vertexValues,
    AccumValue     *accumulators,
    int            *activties,
    F f)
{
    int tid = THREAD_INDEX;
    if (tid >= *workqueueSize) return;
    VertexId srcId = workqueue[tid];
    EdgeId start = vertices[srcId];
    EdgeId end = vertices[srcId + 1];
    EdgeId outdegree = end - start;
    VertexValue srcValue = vertexValues[srcId];

    for (EdgeId e = start; e < end; e ++) {
        // Edge level parallelism, which is exploited by SIMD lanes
        AccumValue accum = f.gather(srcValue, outdegree);
        VertexId dstId = outgoingEdges[e];
        f.reduce(accumulators[dstId], accum);
        activties[dstId] = 1;
    }
}

/**
 * The vertex map kernel.
 *
 * The initial value of `allVerticesInactive` is true. All the active vertices
 * write false to `allVerticesInactive`. When there is no vertex is active,
 * the final value will be false.
 */
template<typename VertexValue,
         typename AccumValue,
         typename F>
__global__
void vertexMapKernel(
    int          verticeCount,
    VertexValue *vertexValues,
    F f)
{
    int tid = THREAD_INDEX;
    if (tid >= verticeCount) return;
    f.update(vertexValues[tid]);  
}

template<typename VertexValue,
         typename AccumValue,
         typename F>
__global__
void vertexFilterKernel(
    int         *activties,
    int          verticeCount,
    VertexValue *vertexValues,
    AccumValue  *accumulators,
    VertexId    *workqueue,
    VertexId    *workqueueSize,
    F f)
{
    int tid = THREAD_INDEX;
    if (tid >= verticeCount) return;
    if (activties[tid] == 0) return;
    // Deactivate at first. Then recover
    activties[tid] = 0; 
    if (f.cond(vertexValues[tid], tid)) {
        f.update(vertexValues[tid], accumulators[tid]);
        activties[tid] = 1;
        VertexId pos = atomicAdd(workqueueSize, 1);
        workqueue[pos] = tid;
    }
}



#endif  // OLIVER_KERNEL_H
