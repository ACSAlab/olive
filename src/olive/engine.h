/**
 * The core components of olive.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-12-20
 * Last Modified: 2014-12-20
 */

#ifndef ENGINE_H
#define ENGINE_H

#include <vector>
#include <iomanip>

#include "common.h"
#include "flexible.h"
#include "partition.h"
#include "partition_strategy.h"
#include "logging.h"

#define INF_COST 0x7fffffff

__global__
void scatterKernel(
    MessageBox<Partition::Stub> *inbox,
    int *local_levels,
    int *workset)
{
    int tid = THREAD_INDEX;
    if (tid >= inbox->length) return;
    VertexId inNode = inbox->buffer[tid].localId;

    if (local_levels[inNode] == INF_COST) {
        int remotelevel = reinterpret_cast<int> (inbox->buffer[tid].message);
        local_levels[inNode] = remotelevel + 1;
        workset[inNode] = 1;
    }
}

__global__
void compactKernel(
    int *workset,
    size_t n,
    VertexId *workqueue,
    size_t *workqueueSize)
{
    int tid = THREAD_INDEX;
    if (tid >= n) return;
    if (workset[tid] == 1) {
        workset[tid] = 0;
        size_t offset = atomicAdd(reinterpret_cast<unsigned long long *> (workqueueSize), 1);
        workqueue[offset] = tid;
    }
}

__global__
void expandKernel(
    PartitionId thisPid,
    const EdgeId *vertices,
    const Partition::Vertex *edges,
    MessageBox<Partition::Stub> *outboxes,
    int *workset,
    VertexId *workqueue,
    int n,
    int *local_levels)
{
    int tid = THREAD_INDEX;
    if (tid >= n) return;
    VertexId outNode = workqueue[tid];
    EdgeId first = vertices[outNode];
    EdgeId last = vertices[outNode + 1];
    for (EdgeId edge = first; edge < last; edge ++) {
        PartitionId pid = edges[edge].partitionId;
        if (pid == thisPid) {  // In this partition
            VertexId inNode = edges[edge].localId;
            if (local_levels[inNode] == INF_COST) {
                local_levels[inNode] = local_levels[outNode] + 1;
                workset[inNode] = 1;
            }
        } else {  // In remote partition
            size_t offset = atomicAdd(reinterpret_cast<unsigned long long *> (&outboxes[pid].length), 1);
            Partition::Stub stub;
            stub.localId = edges[edge].localId;
            stub.message = reinterpret_cast<void *>(local_levels[outNode]);
            outboxes[pid].buffer[offset] = stub;
        }
    }
}


/** Per-node state. */
typedef struct {
    int *levels;
    int *levels_h;
} bfs_state_t;

// typedef struct {
//     int *levels;
//     int *levels_h;
//     Partition::Vertex source;
// } bfs_global_state_t;

// static bfs_global_state_t state_g = {NULL, NULL, {0, 0}};

// Allocate the state for each partition
void init_bfs(Partition &part) {
    bfs_state_t *state = new bfs_state_t;

    VertexId n = part.vertexCount;

    state->levels_h = new int[n];
    for (int i = 0; i < n; i++) {
        state->levels_h[i] = INF_COST;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state->levels), sizeof(int) * n));
    CUDA_CHECK(H2D(state->levels, state->levels_h, sizeof(int) * n));

    if (0 == part.partitionId) {
        VertexId vid = 0;
        state->levels_h[vid] = 0;
        CUDA_CHECK(H2D(state->levels + vid, state->levels_h + vid, sizeof(int)));
        part.workqueue.set(0, vid);
        *(part.workqueueSize) = 1;
        CUDA_CHECK(H2D(part.workqueueSizeDevice, part.workqueueSize, sizeof(int)));
    }
    part.algoState = state;
}

class Engine {
private:
    int supersteps;
    bool terminate;
    std::vector<Partition> partitions;

    double initAlgoStateTime;

public:
    /**
     * [config description]
     * @param subgraphs [description]
     */
    void init(const char *path, int numParts) {
        util::enableAllPeerAccess();
        flex::Graph<int, int> graph;
        graph.fromEdgeListFile(path);
        RandomEdgeCut random ;
        auto subgraphs = graph.partitionBy(random, numParts);
        partitions.resize(subgraphs.size());
        for (int i = 0; i < subgraphs.size(); i++) {
            partitions[i].fromSubgraph(subgraphs[i]);
        }
    }

    ~Engine() {
        LOG(INFO) << "initAlgoStateTime: " << std::setprecision(3) << initAlgoStateTime << "ms";
    }


    /** allocate local states for each partition */
    void initAlgoStates() {
        double startTime = util::currentTimeMillis();
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            init_bfs(partitions[i]);
        }
        initAlgoStateTime = util::currentTimeMillis() - startTime;
    }

    /** Run the engine until all partition vote to quit */
    void run() {
        initAlgoStates();
        for (int i = 0; i < partitions.size(); i++) {
            assert(partitions[i].algoState);
        }
        supersteps = 0;
        while (true) {
            terminate = true;
            superstep();
            supersteps += 1;

            if (terminate) break;
        }
    }

    void superstep() {
        double startTime = util::currentTimeMillis();
        //////////////////////////// Computation stage /////////////////////////
        // Before launching the kernel, scatter the local state according
        // to the inbox's stubs.
        if (supersteps > 0) {
            for (int i = 0; i < partitions.size(); i++) {
                for (int j = 0; j < partitions.size(); j++) {
                    if (partitions[i].inboxes[j].length == 0) continue;
                    CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                    auto config = util::kernelConfig(partitions[i].inboxes[j].length);
                    scatterKernel <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                        &partitions[i].inboxes[j],
                        reinterpret_cast<bfs_state_t *> (partitions[i].algoState)->levels,
                        partitions[i].workset.elemsDevice);
                }
            }

            // Compacting the workset back to the workqueue
            // Clear the queue before generating it
            for (int i = 0; i < partitions.size(); i++) {
                *partitions[i].workqueueSize = 0;
                CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                CUDA_CHECK(H2D(partitions[i].workqueueSizeDevice,
                               partitions[i].workqueueSize, sizeof(size_t)));
                auto config = util::kernelConfig(partitions[i].vertexCount);
                compactKernel <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                    partitions[i].workset.elemsDevice,
                    partitions[i].vertexCount,
                    partitions[i].workqueue.elemsDevice,
                    partitions[i].workqueueSizeDevice);
            }
        }

        // Transfer all the workqueueSize to host to judge whether terminate.
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(D2H(partitions[i].workqueueSize,
                           partitions[i].workqueueSizeDevice, sizeof(size_t)));
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " work queue size=" << *partitions[i].workqueueSize;
            if (*partitions[i].workqueueSize != 0) {
                terminate = false;
            }
        }

        bool * expandLaunched = new bool[partitions.size()];
        for (int i = 0; i < partitions.size(); i++) {
            expandLaunched[i] = true;
        }

        // In each super step, launches the expand kernel for all partitions.
        // The computation kernel is launched in the stream 1.
        // Jump over the partition that has no work to perform.
        for (int i = 0; i < partitions.size(); i++) {
            if (*partitions[i].workqueueSize == 0) {
                expandLaunched[i] = false;
                continue;
            }
            LOG(INFO) << "Partition" << partitions[i].partitionId
                      << " launching expansion kernel on " 
                      << *partitions[i].workqueueSize << " elements";
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventRecord(partitions[i].startEvent, partitions[i].streams[1]));
            auto config = util::kernelConfig(*partitions[i].workqueueSize);
            expandKernel <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                partitions[i].partitionId,
                partitions[i].vertices.elemsDevice,
                partitions[i].edges.elemsDevice,
                partitions[i].outboxes,
                partitions[i].workset.elemsDevice,
                partitions[i].workqueue.elemsDevice,
                *partitions[i].workqueueSize,
                reinterpret_cast<bfs_state_t *> (partitions[i].algoState)->levels);
            CUDA_CHECK(cudaEventRecord(partitions[i].endEvent, partitions[i].streams[1]));
        }

        ///////////////////////// Communication stage //////////////////////////
        // All-to-all message box transferring.
        // To satisfy the dependency, the asynchronous copy can only be launched
        // in the same stream as that of the source partition.
        // The copy will be launched strictly after the source partition has
        // got the all computation done, and got all outboxes ready.
        for (int i = 0; i < partitions.size(); i++) {
            for (int j = i + 1; j < partitions.size(); j++) {
                partitions[i].inboxes[j].recvMsgs(partitions[j].outboxes[i],
                                                  partitions[j].streams[1]);
                partitions[j].inboxes[i].recvMsgs(partitions[i].outboxes[j],
                                                  partitions[i].streams[1]);
                LOG(DEBUG) << partitions[j].outboxes[i].length << " msgs "
                           << "from Partition" << j << " to " << i;
                LOG(DEBUG) << partitions[i].outboxes[j].length << " msgs "
                           << "from Partition" << i << " to " << j;
            }
        }
        ///////////////////////// Synchronization stage ////////////////////////
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            // CUDA_CHECK(cudaStreamSynchronize(partitions[i].streams[0]));
            CUDA_CHECK(cudaStreamSynchronize(partitions[i].streams[1]));
        }
        // Swaps the inbox for each partition before the next super step
        // begins. So that each partition can process up-to-date data.
        // Each partition has ``-1 inboxes to swap.
        for (int i = 0; i < partitions.size(); i++) {
            for (int j = 0; j < partitions.size(); j++) {
                if (i == j) continue;
                partitions[i].inboxes[j].swapBuffers();
            }
        }
        // Collect the execution time for each computing kernel.
        // Choose the lagging one to represent the computation time.
        double totalTime = util::currentTimeMillis() - startTime;
        double compTime = 0.0;
        float kernelTime;
        for (int i = 0; i < partitions.size(); i++) {
            if (!expandLaunched[i]) continue;
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventElapsedTime(&kernelTime,
                                            partitions[i].startEvent,
                                            partitions[i].endEvent));
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << ": comp=" << std::setprecision(2) << kernelTime << "ms";
            if (kernelTime > compTime) compTime = kernelTime;
        }

        double commTime = totalTime - compTime;
        LOG(INFO) << "Superstep" << supersteps
                  << ": total=" << std::setprecision(3) << totalTime << "ms"
                  << ", comp=" << std::setprecision(2) << compTime / totalTime
                  << ", comm=" << std::setprecision(2) << commTime / totalTime;
    }

    // // Aggregate the local states on each partition to global states
    // void aggregate() {
    //     double startTime = util::currentTimeMillis();
    //     for (auto part : partitions) {
    //         config.aggregate(part);
    //     }
    //     double aggrTime = util::currentTimeMillis() - startTime;
    //     LOG(INFO) << "Aggregagte" << std::setprecision(3) << aggrTime
    // }
};

#endif  // ENGINE_H
