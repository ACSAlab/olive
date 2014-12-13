/**
 * Partition.
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
#include "message_box.h"

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
 * in a graph partition. e.g. The ids are continuous from 0 to `vertices`-1;
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
        explicit Vertex(PartitionId pid, VertexId id_) {
            partitionId = pid;
            id = id_;
        }
    };

    /**
     * Stub information sending to a remote vertex.
     * @note `id` here is .
     */
    class Stub {
     public:
        VertexId  id;       /** Specifying the remote vertex by its local id. */
        void *    message;  /** Content of the message. */
    };

    /** Partition identification. Obtained via a pass-in subgraph */
    PartitionId    partitionId;

    /** Number of partitions. Obtained via a pass-in subgraph */
    PartitionId    numParts;

    /** The device this partition binds to */
    int            deviceId;

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
     * Messages sending to remote vertices are inserted into corresponding 
     * `outboxes`. The number of outboxes is equal to the number of partitions
     * in the graph, which is got in runtime.
     * If the vertex is in remote partition, then a message is inserted into
     * the corresponding outbox. For the convenience, there is an empty outbox
     * reserved for the local partition (do not allocate memory).
     * e.g. for partition 2, outboxes[0/1/3] is effective.
     */
    MessageBox<Stub> * outboxes;

    /**
     * Messages received from remote vertices.
     */
    MessageBox<Stub> * inboxes;

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

    /** Constructor */
    Partition(): partitionId(0), deviceId(0), numParts(1),
        outboxes(NULL), inboxes(NULL) {}

    /**
     * Initializing a partition from a subgraph in flexible representation. 
     * By default, the partition is bound to device 0.
     * 
     * The subgraph marks the vertices with a global id. Records the original
     * ids in `globalIds`.
     * 
     *
     * TODO(onesuper): more complicated partition-to-device assignment.
     */
    void fromSubgraph(const flex::Graph<int, int> &subgraph) {
        partitionId = subgraph.partitionId;
        numParts = subgraph.numParts;
        vertices.reserve(subgraph.nodes()+1, deviceId);
        if (subgraph.edges() > 0)
            edges.reserve(subgraph.edges(), deviceId);
        if (subgraph.nodes() > 0)
            globalIds.reserve(subgraph.nodes(), deviceId);

        // `toLocal` maps the global id to local. Used to create remote vertex.
        std::map<VertexId, VertexId> toLocal;
        VertexId localId     = 0;
        VertexId vertexCount = 0;
        EdgeId   edgeCount   = 0;
        for (auto v : subgraph.vertices) {
            globalIds[localId++] = v.id;
            toLocal.insert(std::pair<VertexId, VertexId>(v.id, localId));
        }
        for (auto v : subgraph.vertices) {
            vertices[vertexCount++] = edgeCount;
            for (auto e : v.outEdges) {
                if (subgraph.hasVertex(e.id)) {  // In this partition
                    edges[edgeCount++] = Vertex(partitionId, toLocal[e.id]);
                } else {                         // In remote partition
                    auto it = subgraph.ghostVertices.find(e.id);
                    assert(it != subgraph.ghostVertices.end());
                    edges[edgeCount++] = Vertex(it->second.first,
                                                it->second.second);
                }
            }
        }
        vertices[vertexCount] = edgeCount;
        assert(localId == subgraph.nodes());
        assert(vertexCount == subgraph.nodes());
        assert(edgeCount == subgraph.edges());
        vertices.cache();
        edges.cache();
        globalIds.cache();

        initMessageBoxes(subgraph);

        // Sets up the CUDA resources.
        deviceId = partitionId % 2;
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaStreamCreate(&streams[0]));
        CUDA_CHECK(cudaStreamCreate(&streams[1]));
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&endEvent));
    }

    /** Destructor **/
    ~Partition() {
        if (outboxes) delete[] outboxes;
        if (inboxes)  delete[] inboxes;
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaStreamDestroy(streams[0]));
        CUDA_CHECK(cudaStreamDestroy(streams[1]));
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(endEvent));
    }

 private:
    /**
     * Initializing `outboxes` and `inboxes` according to the topological
     * information of the subgraph, since we have to allocate memory before-hand
     * on GPUs.
     *
     * The size of `outbox` for a partition is equal to the number of those
     * outgoing edges connecting to this remote partition. And the size of
     * `inbox` for a partition is equal to the number of those incoming edges
     * coming from this remote partition.
     *
     * The message boxes is allocated at the maximum size when every vertex 
     * wants to send a message to its remote neighbors.
     *
     * @note Duplication is allowed when counting incoming/outgoing edges, since
     * it is possible that more than one vertex in local partition send messages
     * to the same remote vertex.
     */
    void initMessageBoxes(const flex::Graph<int, int> &subgraph) {
        int * outgoingEdges = new int[numParts]();
        int * incomingEdges = new int[numParts]();
        outboxes = new MessageBox<Stub>[numParts];
        inboxes = new MessageBox<Stub>[numParts];
        for (auto v : subgraph.vertices) {
            for (auto e : v.outEdges) {
                if (!subgraph.hasVertex(e.id)) {
                    auto it = subgraph.ghostVertices.find(e.id);
                    assert(it != subgraph.ghostVertices.end());
                    PartitionId parTo = it->second.first;
                    assert(parTo < numParts);
                    outgoingEdges[parTo]++;
                }
            }
            for (auto e : v.inEdges) {
                if (!subgraph.hasVertex(e.id)) {
                    auto it = subgraph.ghostVertices.find(e.id);
                    assert(it != subgraph.ghostVertices.end());
                    PartitionId parFrom = it->second.first;
                    assert(parFrom < numParts);
                    incomingEdges[parFrom]++;
                }
            }
        }
        assert(outgoingEdges[partitionId] == 0);
        assert(incomingEdges[partitionId] == 0);
        for (PartitionId i = 0; i < numParts; i++) {
            if (i == partitionId) continue;
            if (outgoingEdges[i] > 0)
                outboxes[i].reserve(outgoingEdges[i]);
            if (incomingEdges[i] > 0)
                inboxes[i].reserve(incomingEdges[i]);
        }
        delete[] outgoingEdges;
        delete[] incomingEdges;
    }
};

#endif  // PARTITION_H
