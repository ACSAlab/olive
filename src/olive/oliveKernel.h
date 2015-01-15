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


template<typename VertexValue,
         typename AccumValue,
         typename EdgeFunction>
__global__
void edgeGatherDenseKernel(
    PartitionId  thisPid,
    const EdgeId *vertices,
    const Vertex *edges,
    MessageBox< VertexMessage<AccumValue> > *outboxes,
    VertexId vertexCount,
    VertexValue *vertexValues,
    AccumValue *accumulators,
    EdgeFunction edgeFunc)
{
    int tid = THREAD_INDEX;
    if (tid >= vertexCount) return;
    VertexValue srcValue = vertexValues[tid];
    EdgeId first = vertices[tid];
    EdgeId last = vertices[tid + 1];
    EdgeId outdegree = last - first + 1;

    for (EdgeId edge = first; edge < last; edge ++) {
        PartitionId dstPid = edges[edge].partitionId;
        // Edge level parallelism, which is exploited by SIMD lanes
        AccumValue accum = edgeFunc.gather(srcValue, outdegree);

        if (dstPid == thisPid) {  // In this partition
            VertexId dstId = edges[edge].localId;
            edgeFunc.reduce(accumulators[dstId], accum);
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
 * The CUDA kernel for expanding vertices in the work queue.
 */
template<typename VertexValue,
         typename AccumValue,
         typename EdgeFunction>
__global__
void edgeGatherSparseKernel(
    PartitionId  thisPid,
    const EdgeId *vertices,
    const Vertex *edges,
    MessageBox< VertexMessage<AccumValue> > *outboxes,
    const VertexId *workqueue,
    int queueSize,
    VertexValue *vertexValues,
    AccumValue *accumulators,
    EdgeFunction edgeFunc)
{
    int tid = THREAD_INDEX;
    if (tid >= queueSize) return;
    VertexId srcId = workqueue[tid];
    VertexValue srcValue = vertexValues[srcId];
    EdgeId first = vertices[srcId];
    EdgeId last = vertices[srcId + 1];
    EdgeId outdegree = last - first + 1;

    for (EdgeId edge = first; edge < last; edge ++) {
        PartitionId dstPid = edges[edge].partitionId;
        // Edge level parallelism, which is exploited by SIMD lanes
        AccumValue accum = edgeFunc.gather(srcValue, outdegree);

        if (dstPid == thisPid) {  // In this partition
            VertexId dstId = edges[edge].localId;
            edgeFunc.reduce(accumulators[dstId], accum);
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
 * The edgeMap and edgeFilter reuse the same code piece and differentiate
 * by a template flag `VertexFiltered`.
 */
template<typename AccumValue, typename EdgeFunction>
__global__
void edgeScatterKernel(
    const MessageBox< VertexMessage<AccumValue> > &inbox,
    AccumValue *accumulators,
    EdgeFunction edgeFunc)
{
    int tid = THREAD_INDEX;
    if (tid >= inbox.length) return;
    VertexId dstId = inbox.buffer[tid].receiverId;
    AccumValue accum = inbox.buffer[tid].value;
    edgeFunc.reduce(accumulators[dstId], accum);
}

/**
 * The vertex map kernel.
 *
 * @param f   A user-defined functor to update the vertex state.
 */
template<typename VertexValue,
         typename AccumValue,
         typename VertexFunction>
__global__
void vertexMapKernel(
    VertexValue *vertexValues,
    int verticeCount,
    AccumValue *accumulators,    
    VertexFunction f)
{
    int tid = THREAD_INDEX;
    if (tid >= verticeCount) return;
    f(vertexValues[tid], accumulators[tid]);
}

/**
 * The vertex filter kernel.
 *
 * @param id  The id of the vertex to filter out.
 * @param f   A user-defined functor to update the vertex state.
 */
template<typename VertexValue,
         typename AccumValue,
         typename VertexFunction>
__global__
void vertexFilterKernel(
    int *workset,
    int vertexCount,
    VertexId *workqueue,
    VertexId *workqueueSize,
    VertexValue *vertexValues,
    AccumValue *accumulators,
    VertexFunction f)
{
    int tid = THREAD_INDEX;
    if (tid >= vertexCount) return;

    f.update(vertexValues[tid], accumulators[tid]);

    if (f.cond(vertexValues[tid]) == true) {
        VertexId offset = atomicAdd(reinterpret_cast<unsigned long long *>
                                   (workqueueSize), 1);
        workqueue[offset] = tid;
    }
}



#endif  // OLIVE_KERNEL_H
