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

#ifndef OLIVE_KERNEL_H
#define OLIVE_KERNEL_H

#include "common.h"


/**
 * The CUDA kernel for expanding vertices in the work queue.
 */
template<typename VertexValue,
         typename AccumValue,
         typename F>
__global__
void edgeGatherKernel(
    PartitionId     thisPid,
    const VertexId *workqueue,
    const VertexId *workqueueSize,
    const EdgeId   *vertices,
    const Vertex   *edges,
    VertexValue    *vertexValues,
    AccumValue     *accumulators,
    int            *activties,
    MessageBox< VertexMessage<AccumValue> > *outboxes,
    F f)
{
    int tid = THREAD_INDEX;
    if (tid >= *workqueueSize) return;
    VertexId srcId = workqueue[tid];
    VertexValue srcValue = vertexValues[srcId];
    EdgeId first = vertices[srcId];
    EdgeId last = vertices[srcId + 1];
    EdgeId outdegree = last - first + 1;

    for (EdgeId edge = first; edge < last; edge ++) {
        PartitionId dstPid = edges[edge].partitionId;
        // Edge level parallelism, which is exploited by SIMD lanes
        AccumValue accum = f.gather(srcValue, outdegree);

        if (dstPid == thisPid) {  // In this partition
            VertexId dstId = edges[edge].localId;
            f.reduce(accumulators[dstId], accum);
            activties[dstId] = 1;
        } else {  // In remote partition
            VertexMessage<AccumValue> msg;
            msg.receiverId = edges[edge].localId;
            msg.value      = accum;
            size_t offset = atomicAdd(reinterpret_cast<unsigned long long *>
                                      (&outboxes[dstPid].length), 1);
            outboxes[dstPid].buffer[offset] = msg;
        }
    }
}

/**
 * The CUDA kernel for scattering messages to local vertex values.
 * If a node receive any message from other partition, the vertex will be
 * marked as active.
 */
template<typename AccumValue, typename F>
__global__
void edgeScatterKernel(
    const MessageBox< VertexMessage<AccumValue> > &inbox,
    AccumValue *accumulators,
    int *activties,
    F f)
{
    int tid = THREAD_INDEX;
    if (tid >= inbox.length) return;
    VertexId dstId = inbox.buffer[tid].receiverId;
    AccumValue accum = inbox.buffer[tid].value;
    f.reduce(accumulators[dstId], accum);
    activties[dstId] = 1;
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

    // Deactivate anyway, since we only want the vertex with changed
    // accumulator to be activated in the vertex phase.
    activties[tid] = 0;  

    if (f.cond(vertexValues[tid])) {
        f.update(vertexValues[tid], accumulators[tid]);

        // if the local state is modified,  activate it
        activties[tid] = 1;        
        VertexId pos = atomicAdd(workqueueSize, 1);
        workqueue[pos] = tid;
    }
}

template<typename VertexValue,
         typename AccumValue,
         typename F>
__global__
void vertexFilterKernel(
    int         *activties,
    int          verticeCount,
    VertexValue *vertexValues,
    VertexId    *workqueue,
    VertexId    *workqueueSize,
    F f)
{
    int tid = THREAD_INDEX;
    if (tid >= verticeCount) return;
    if (f.cond(vertexValues[tid])) {
        f.update(vertexValues[tid]);
        activties[tid] = 1;
        VertexId pos = atomicAdd(workqueueSize, 1);
        workqueue[pos] = tid;
    }
}



#endif  // OLIVE_KERNEL_H
