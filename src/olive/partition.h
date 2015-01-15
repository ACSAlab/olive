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
#include <iomanip>

#include "grd.h"
#include "flexible.h"
#include "utils.h"
#include "logging.h"
#include "messageBox.h"

/**
 * If a vertex's `partitionId` equals the partition's `partitionId`, then it
 * refers to a local vertex. Otherwise `localId` refers to a remote one.
 */
class Vertex {
public:
    PartitionId  partitionId;
    VertexId     localId;

    Vertex(): partitionId(0), localId(0) {}
    explicit Vertex(PartitionId pid, VertexId id): partitionId(pid), localId(id) {}
};

/**
 * Message information sending to a remote vertex.
 * @note `id` here is .
 */
template<typename MessageValue>
class VertexMessage {
public:
    /** Specifying the remote vertex by its local id. */
    VertexId        receiverId;
    /** Content of the message. */
    MessageValue    value;

    /** reserved for the printing on CPU. */
    void print() {
        printf("%d:%lld  ", receiverId, reinterpret_cast<long long int> (value));
    }
};

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
template<typename VertexValue, typename AccumValue>
class Partition {
public:

    /** Partition identification. Obtained via a pass-in subgraph */
    PartitionId    partitionId;

    /** Number of partitions. Obtained via a pass-in subgraph */
    PartitionId    numParts;

    /** The device this partition binds to */
    int            deviceId;

    /** Record the edge and vertex number of each partition. */
    VertexId       vertexCount;
    EdgeId         edgeCount;

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
     * Partition-wise vertex state values.
     */
    GRD<VertexValue> vertexValues;

    /**
     * For each vertex an accumulator is cached to perform computation.
     */
    GRD<AccumValue> accumulators;

    /**
     * Use a bitmap to represent the working set.
     */
    GRD<int>       workset;

    /**
     * Use a queue to keep the work complexity low
     */
    GRD<VertexId>  workqueue;
    VertexId      *workqueueSize;       /** queue size */
    VertexId      *workqueueSizeDevice; /** queue size device */

    /**
     * Messages sending to remote vertices are inserted into corresponding
     * `outboxes`. The number of outboxes is equal to the number of partitions
     * in the graph, which is got in runtime.
     * If the vertex is in remote partition, then a message is inserted into
     * the corresponding outbox. For the convenience, there is an empty outbox
     * reserved for the local partition (do not allocate memory).
     * e.g. for partition 2, outboxes[0/1/3] is effective.
     */
    MessageBox< VertexMessage<AccumValue> > *outboxes;

    /**
     * Messages received from remote vertices.
     */
    MessageBox< VertexMessage<AccumValue> > *inboxes;

    /**
     * Enables overlapped communication and computation.
     * The computation and communication within the same stream is sequential.
     * And different streams can overlap.
     */
    cudaStream_t    streams[2];

    /**
     * Measures the execution time of a certain kernel.
     */
    cudaEvent_t     startEvents[4];
    cudaEvent_t     endEvents[4];

    /**
     * Constructor
     * It is important to give a NULL value to avoid delete a effective pointer.
     */
    Partition() {
        deviceId = -1;
        partitionId = 0;
        numParts = 0;
        outboxes = NULL,
        inboxes = NULL;
        workqueueSize = NULL;
        workqueueSizeDevice = NULL;
        streams[0] = NULL;
        streams[1] = NULL;
        for (int i = 0; i < 4; i++) {
            startEvents[i] = NULL;
            endEvents[i] = NULL;
        }
    }

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
        // TODO(onesuper): change later
        deviceId = partitionId % 2;
        vertexCount = subgraph.nodes();
        edgeCount = subgraph.edges();
        // Only reserve memory if the graph has at least one edge/node
        if (edgeCount == 0 || vertexCount == 0) {
            LOG(WARNING) << "Parition" << partitionId << " #vertices= "
            << vertexCount << ", #edges= " << edgeCount;
        }

        double startTime = util::currentTimeMillis();
        util::Stopwatch stopwatch;
        stopwatch.start();

        // Sets up the CUDA resources.
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaStreamCreate(&streams[0]));
        CUDA_CHECK(cudaStreamCreate(&streams[1]));

        for (int i = 0; i < 4; i++) {
            CUDA_CHECK(cudaEventCreate(&startEvents[i]));
            CUDA_CHECK(cudaEventCreate(&endEvents[i]));
        }

        // Allocate the memory for the buffers on CPU and GPU
        vertices.reserve(vertexCount + 1, deviceId);
        edges.reserve(edgeCount, deviceId);
        globalIds.reserve(vertexCount, deviceId);
        vertexValues.reserve(vertexCount, deviceId);
        accumulators.reserve(vertexCount, deviceId);
        workqueue.reserve(vertexCount, deviceId);
        workset.reserve(vertexCount, deviceId);
        workqueueSize = static_cast<VertexId *> (malloc(sizeof(VertexId)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **> (&workqueueSizeDevice),
                              sizeof(VertexId)));

        double allocTime = stopwatch.elapsedMillis();

        // Maps the global ids to local ids. Used to create remote vertex.
        std::map<VertexId, VertexId> toLocal;
        VertexId localId = 0;
        for (auto v : subgraph.vertices) {
            globalIds[localId] = v.id;
            toLocal.insert(std::pair<VertexId, VertexId>(v.id, localId));
            localId += 1;
        }
        assert(localId == vertexCount);
        // Traverses all nodes and out-going edges to set up CSR data.
        VertexId vertexCursor = 0;
        EdgeId   edgeCursor   = 0;
        for (auto v : subgraph.vertices) {
            vertices[vertexCursor] = edgeCursor;
            for (auto e : v.outEdges) {
                Vertex dst;
                if (subgraph.hasVertex(e.vertexId)) {  // In this partition
                    dst.partitionId = partitionId;
                    dst.localId = toLocal[e.vertexId];
                } else {                         // In remote partition
                    auto it = subgraph.ghostVertices.find(e.vertexId);
                    assert(it != subgraph.ghostVertices.end());
                    dst.partitionId = it->second.first;
                    dst.localId = it->second.second;
                }
                edges[edgeCursor] = dst;
                edgeCursor += 1;
            }
            vertexCursor += 1;
        }
        vertices[vertexCursor] = edgeCursor;  // Close the edge
        assert(vertexCount == vertexCursor);
        assert(edgeCount == edgeCursor);
        double indexTime = stopwatch.elapsedMillis();

        // Transfer all the buffers to GPU.
        vertices.cache();
        edges.cache();
        globalIds.cache();
        *workqueueSize = 0;
        CUDA_CHECK(H2D(workqueueSizeDevice, workqueueSize, sizeof(VertexId)));
        workset.allTo(0);
        double cacheTime = stopwatch.elapsedMillis();

        // Initialize the message boxes accordingly.
        initMessageBoxes(subgraph);
        double msgboxTime = stopwatch.elapsedMillis();

        double totalTime = util::currentTimeMillis() - startTime;
        LOG(INFO) << "It took " << std::setprecision(3) << totalTime
                  << "ms to land partition" << partitionId
                  << "(V=" << vertexCount << ", E=" << edgeCount
                  << ") on device " << deviceId << std::fixed
                  << ", Alloc=" << std::setprecision(1) << allocTime / totalTime
                  << ", Index=" << std::setprecision(1) << indexTime / totalTime
                  << ", Cache=" << std::setprecision(1) << cacheTime / totalTime
                  << ", MsgBox=" << std::setprecision(1) << msgboxTime / totalTime;
    }

    /** Destructor **/
    ~Partition() {
        if (deviceId >= 0) CUDA_CHECK(cudaSetDevice(deviceId));
        // if (outboxes)   delete[] outboxes;
        // if (inboxes)    delete[] inboxes;
        if (outboxes) CUDA_CHECK(cudaFreeHost(outboxes));
        if (inboxes) CUDA_CHECK(cudaFreeHost(inboxes));
        if (streams[0]) CUDA_CHECK(cudaStreamDestroy(streams[0]));
        if (streams[1]) CUDA_CHECK(cudaStreamDestroy(streams[1]));
        if (workqueueSize) free(workqueueSize);
        if (workqueueSizeDevice) CUDA_CHECK(cudaFree(workqueueSizeDevice));
        for (int i = 0; i < 4; i++) {
            if (startEvents[i]) CUDA_CHECK(cudaEventDestroy(startEvents[i]));
            if (endEvents[i])   CUDA_CHECK(cudaEventDestroy(endEvents[i]));
        }
    }

    // Returns the address of a neighbors' state by giving a Vertex value. if
    // the vertex is in local partition, returns the address in `localState`.
    // Otherwise, returns the address in outbox to that remote partition.
    //
    // Note the message buffer an be accessed by all CUDA contexts.
    // template<typename T>
    // __device__
    // inline T* getNeighborState(Vertex ngh , T* localState) {
    //     if (ngh.partitionId != partitionId) {
    //         T* address = (T*) outboxes[ngh.partitionId].buffer0
    //         return &address[ngh.localId];
    //     }
    //     return &localState[ngh.localId]
    // }

private:
    /**
     * The number of the outboxes or inboxes depends on the `numPart`.
     *
     * Initializing `outboxes` and `inboxes` according to the topological
     * information of the subgraph, since we have to allocate memory before-hand
     * on GPUs.
     *
     * The size of each `outbox` for a partition is equal to the number of those
     * outgoing edges connecting to this remote partition. The message boxes is
     * allocated at the maximum size when every vertex in the graph has a message
     * sent to its remote neighbors.
     *
     * The size of each `inbox` for a partition is equal to the number of those
     * incoming edges coming from this remote partition.
     *
     * @note Duplication is allowed when counting incoming/outgoing edges, since
     * it is possible that more than one vertex in local partition send messages
     * to the same remote vertex.
     */
    void initMessageBoxes(const flex::Graph<int, int> &subgraph) {
        int *outgoingEdges = new int[numParts]();
        int *incomingEdges = new int[numParts]();

        // The pointers are allocated in pinned memory so that they can be
        // accessed as outboxes[i] in any CUDA contexts.
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **> (&outboxes),
                                  sizeof(MessageBox< VertexMessage<AccumValue> >) * numParts,
                                  cudaHostAllocPortable));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **> (&inboxes),
                                  sizeof(MessageBox< VertexMessage<AccumValue> >) * numParts,
                                  cudaHostAllocPortable));

        for (auto v : subgraph.vertices) {
            for (auto e : v.outEdges) {
                if (!subgraph.hasVertex(e.vertexId)) {
                    auto it = subgraph.ghostVertices.find(e.vertexId);
                    assert(it != subgraph.ghostVertices.end());
                    PartitionId parTo = it->second.first;
                    assert(parTo < numParts);
                    outgoingEdges[parTo]++;
                }
            }
            for (auto e : v.inEdges) {
                if (!subgraph.hasVertex(e.vertexId)) {
                    auto it = subgraph.ghostVertices.find(e.vertexId);
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
