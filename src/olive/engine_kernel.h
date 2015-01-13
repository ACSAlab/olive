/**
 * The CUDA kernel of the engine.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-12-20
 * Last Modified: 2014-12-20
 */

#ifndef ENGINE_KERNEL_H
#define ENGINE_KERNEL_H

#include "common.h"

/**
 * The CUDA kernel for expanding vertices in the work set.
 *
 * @param thisPid      My partition id.
 * @param vertices     Represents the out-going edges.
 * @param edges        Represent the destination vertices.
 * @param outboxes     Outboxes to all remote partitions.
 * @param workset      Bit-set representation of the work set
 * @param workqueue    Queue representation of the work set
 * @param n            The number of the vertices in the `workqueue`.
 * @param vertexValues Buffer for the local vertex states.
 * @param edgeContext  Edge computation context.
 * @param msgContext   Message packing and unpacking functions.
 */
template<typename VertexValue, typename EdgeContext>
__global__
void edgeExpandKernel(
    PartitionId  thisPid,
    const EdgeId *vertices,
    const Vertex *edges,
    MessageBox< VertexMessage<VertexValue> > *outboxes,
    int *workset,
    const VertexId *workqueue,
    int queueSize,
    VertexValue *vertexValues,
    EdgeContext edgeContext)
{
    int tid = THREAD_INDEX;
    if (tid >= queueSize) return;
    VertexId srcId = workqueue[tid];
    EdgeId first = vertices[srcId];
    EdgeId last = vertices[srcId + 1];

    for (EdgeId edge = first; edge < last; edge ++) {
        PartitionId dstPid = edges[edge].partitionId;
        if (dstPid == thisPid) {  // In this partition
            VertexId dstId = edges[edge].localId;
            if (edgeContext.pred(vertexValues[dstId])) {
                vertexValues[dstId] = edgeContext.update(vertexValues[srcId]);
                workset[dstId] = 1;
            }
        } else {  // In remote partition
            VertexMessage<VertexValue> msg;
            msg.receiverId = edges[edge].localId;
            msg.value      = vertexValues[srcId];

            size_t offset = atomicAdd(reinterpret_cast<unsigned long long *> (&outboxes[dstPid].length), 1);
            outboxes[dstPid].buffer[offset] = msg;
        }
    }
}

/**
 * The CUDA kernel for scattering messages to local vertex values.
 * Usually invoked after all-to-all message passing.
 *
 * @param inbox          The message box received from remote partitions.
 *                       Must be accessible from CUDA context.
 * @param vertexValues   The buffer for the vertex values.
 * @param workset        Bit-set representation of the work set.
 * @param edgeContext    The edge computation context.
 * @param messageContext Use to convert the message content to vertex value.
 */
template<typename VertexValue,
         typename EdgeContext>
__global__
void edgeScatterKernel(
    const MessageBox< VertexMessage<VertexValue> > &inbox,
    VertexValue *vertexValues,
    int *workset,
    EdgeContext edgeContext)
{
    int tid = THREAD_INDEX;
    if (tid >= inbox.length) return;

    VertexId dstId = inbox.buffer[tid].receiverId;
    VertexValue newValue = inbox.buffer[tid].value;

    if (edgeContext.pred(vertexValues[dstId])) {
        vertexValues[dstId] = edgeContext.update(newValue);
        workset[dstId] = 1;
    }
}

/**
 * The CUDA kernel for converting the work set to the work queue.
 * Using 32-bit int to represent 1-bit can avoid atomic operations.
 *  
 * @param workset       Buffer for the bit-set-based work set.
 * @param worksetSize   Bit-set length.
 * @param workqueue     Buffer for the queue-based work set.
 * @param workqueueSize Queue length.
 * 
 */
__global__
void edgeCompactKernel(
    int *workset,
    size_t worksetSize,
    VertexId *workqueue,
    size_t *workqueueSize)
{
    int tid = THREAD_INDEX;
    if (tid >= worksetSize) return;
    if (workset[tid] == 1) {
        workset[tid] = 0;
        size_t offset = atomicAdd(reinterpret_cast<unsigned long long *>(workqueueSize), 1);
        workqueue[offset] = tid;
    }
}

/**
 * The vertex map kernel.
 *
 * @param f   A user-defined functor to update the vertex state.
 */
template<typename VertexFunction, typename VertexValue>
__global__
void vertexMapKernel(
    VertexValue *vertexValues,
    int verticeCount,
    VertexFunction f)
{
    int tid = THREAD_INDEX;
    if (tid >= verticeCount) return;
    vertexValues[tid] = f(vertexValues[tid]);
}


/**
 * The vertex filter kernel.
 *
 * @param id  The id of the vertex to filter out.
 * @param f   A user-defined functor to update the vertex state.
 */
template<typename VertexFunction, typename VertexValue>
__global__
void vertexFilterKernel(
    const VertexId *globalIds,
    int vertexCount,
    VertexId FilterId,
    VertexValue *vertexValues,
    VertexFunction f,
    int *workset)
{
    int tid = THREAD_INDEX;
    if (tid >= vertexCount) return;
    if (globalIds[tid] == FilterId) {
        vertexValues[tid] = f(vertexValues[tid]);
        workset[tid] = 1;
    }
}

#endif  // ENGINE_KERNEL_H
