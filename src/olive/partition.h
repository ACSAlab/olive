/**
 * Partition
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-13
 * Last Modified: 2014-11-13
 */


#ifndef PARTITION_H
#define PARTITION_H

#include <vector>
#include <map>
#include <utility>

#include "grd.h"
#include "flexible.h"
#include "utils.h"
#include "logging.h"

/**
 * Managing the resource for each GPU-resident graph partition.
 * 
 * Each graph partition is stored in (Compressed Sparse Row) format for its
 * efficiency. In effect, CSR storage minimizes the memory footprint at the 
 * expense of bringing indirect memory access.
 *
 * Partitions communicate with each other through a buffer-copying scheme: 
 * e.g. One partition copies its out-buffer to the corresponding in-buffer of 
 * the other partition.
 * 
 * @note The original vertex id in original graph is mapped to a new local id
 * in a graph partition. e.g. The ids are continuous from 0 to len(`vertices`)-1;
 *
 */
class Partition {
 public:
    /**
     * If a vertex's `partitionId` equals the partition's `partitionId`, then it
     * refers to a local vertex. Otherwise `id` refers to a remote one. 
     */
    class Vertex {
     public:
        PartitionId  partitionId;
        VertexId     id;
        explicit Vertex(PartitionId pid, VertexId id_) : partitionId(pid), id(id_) {}
    };

    /**
     * Stub information sending to a remote vertex.
     * @note `id` here is .
     */
    class VertexMessage {
     public:
        VertexId  id;
        void *    message;
    };

    /** Partition identification. Obtained via a pass-in subgraph */
    PartitionId    partitionId;

    /** The device this partition binds to */
    PartitionId    deviceId;

    /**
     * Stores the starting indices for querying outgoing edges of local vertices 
     * (vertices that in this partition).
     * e.g., vertices[i] tells where the outgoing edges of vertex `i` are.
     * The number of its outgoing edges is given by vertices[i+1] - vertices[i].
     *
     * @note a.k.a row offsets in matrix terminology.
     */
    GRD<EdgeId>    vertices;

    /**
     * Stores the destination vertex to represent an outgoing edge.
     * It differentiates the boundary edges by `[[DstVertex]].paritionId`.
     * That is if `[[DstVertex]].paritionId` != this->partitionId,
     * the destination vertex is in a remote partition.
     * 
     * @note Boundary edges are those outgoing edges whose destination vertex
     * is in other partitions. When traversing an boundary edge, a message will
     * be push to `outbox`.
     *
     * @note a.k.a column indices in matrix terminology.
     */
    GRD<Vertex>    edges;

    /**
     * Mapping the local linear ids in a partition to original ids when 
     * aggregating the final results.
     */
    GRD<VertexId>  globalIds;

    /**
     * The messages sending to remote vertices are organized as hash tables.
     *
     * TODO: using a double-buffering method.
     */
    std::vector< GRD<VertexMessage> > outboxes;

    /**
     * Messages received from remote vertices.
     */
    std::vector< GRD<VertexMessage> > inboxes;

    /**
     * Send message 
     * @param other The counterpart
     */
    //void send(Partition &other) {}


    /**
     * Enables overlapped communication and computation.
     * e.g., the first stream is used to launch communication operations,
     * while the second one is used to launch computational kernels.
     */
    cudaStream_t    streams[2];

    /**
     * Measures the execution time of a certain kernel.
     */
    cudaEvent_t     startEvent;
    cudaEvent_t     endEvent;

    // void initialize(void) {
    //     CALL_SAFE(cudaSetDevice(deviceId));
    //     CALL_SAFE(cudaStreamCreate(&streams[0]));
    //     CALL_SAFE(cudaStreamCreate(&streams[1]));
    //     CALL_SAFE(cudaEventCreate(&startEvent));
    //     CALL_SAFE(cudaEventCreate(&endEvent));
    // }

    // void finalize(void) {
    //     CALL_SAFE(cudaSetDevice(deviceId));
    //     CALL_SAFE(cudaStreamDestroy(streams[0]));
    //     CALL_SAFE(cudaStreamDestroy(streams[1]));
    //     CALL_SAFE(cudaEventDestroy(startEvent));
    //     CALL_SAFE(cudaEventDestroy(endEvent));
    // }

    /**
     * Initializing a partition from a subgraph in flexible representation. 
     * By default, the partition is bound to device 0.
     *
     * The subgraph marks the vertices with a global id. Records the original
     * ids in `globalIds`.
     * 
     */
    void fromSubgraph(const flex::Graph<int, int> &subgraph) {
        partitionId = subgraph.partitionId;
        deviceId = partitionId % 2;
        vertices.reserve(subgraph.nodes()+1, deviceId);
        edges.reserve(subgraph.edges(), deviceId);
        globalIds.reserve(subgraph.nodes(), deviceId);
        // Building up the `globalIds` and a routing table which maps the global 
        // id to local.
        std::map<VertexId, VertexId> routingToLocal;
        VertexId localId = 0;
        for (auto v : subgraph.vertices) {
            globalIds[localId++] = v.id;
            routingToLocal.insert(std::pair<VertexId, VertexId>(v.id, localId));
        }
        VertexId vertexCount = 0;
        EdgeId   edgeCount   = 0;
        for (auto v : subgraph.vertices) {
            vertices[vertexCount++] = edgeCount;
            for (auto e : v.outEdges) {
                if (subgraph.hasVertex(e.id)) {  // In this partition
                    edges[edgeCount++] = Vertex(partitionId, routingToLocal[e.id]);
                } else {                         // In remote partition
                    auto it = subgraph.ghostVertices.find(e.id);
                    assert(it != subgraph.ghostVertices.end());
                    edges[edgeCount++] = Vertex(it->second.first, it->second.second);
                }
            }
        }
        vertices[vertexCount] = edgeCount;
        assert(vertexCount == subgraph.nodes());
        assert(edgeCount == subgraph.edges());
        vertices.cache();
        edges.cache();
        globalIds.cache();
    }
};



#endif  // PARTITION_H 