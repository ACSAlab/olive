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



/** Per-node state. */
typedef struct {
    int *levels;
    int *levels_h;
} bfs_state_t;

static bfs_state_t state_g = {NULL, NULL};


template<typename MessageValue>
__global__
void scatterKernel(
    MessageBox< VertexMessage<MessageValue> > *inbox,
    void *algoState,
    int *workset)
{
    int tid = THREAD_INDEX;
    if (tid >= inbox->length) return;

    VertexId inNode = inbox->buffer[tid].receiverId;
    int remotelevel = inbox->buffer[tid].value;

    // printf("inNode=%d, levels=%d\n", inNode, local_levels[inNode]);

    bfs_state_t *state = reinterpret_cast<bfs_state_t *> (algoState);

    if (state->levels[inNode] == INF_COST) {
        state->levels[inNode] = remotelevel + 1;
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

template<typename MessageValue>
__global__
void expandKernel(
    PartitionId thisPid,
    const EdgeId *vertices,
    const Vertex *edges,
    MessageBox< VertexMessage<MessageValue> > *outboxes,
    int *workset,
    VertexId *workqueue,
    int n,
    void *algoState)
{
    int tid = THREAD_INDEX;
    if (tid >= n) return;
    VertexId outNode = workqueue[tid];
    EdgeId first = vertices[outNode];
    EdgeId last = vertices[outNode + 1];

    bfs_state_t *state = reinterpret_cast<bfs_state_t *> (algoState);

    for (EdgeId edge = first; edge < last; edge ++) {
        PartitionId pid = edges[edge].partitionId;
        if (pid == thisPid) {  // In this partition
            VertexId inNode = edges[edge].localId;
            if (state->levels[inNode] == INF_COST) {
                state->levels[inNode] = state->levels[outNode] + 1;
                workset[inNode] = 1;
            }
        } else {  // In remote partition
            size_t offset = atomicAdd(reinterpret_cast<unsigned long long *> (&outboxes[pid].length), 1);
            VertexMessage<MessageValue> msg;
            msg.receiverId = edges[edge].localId;
            msg.value = state->levels[outNode];
            outboxes[pid].buffer[offset] = msg;
        }
    }
}

// Allocate the state for each partition
void init_bfs(Partition<int> &part) {
    bfs_state_t *state;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&state), sizeof(bfs_state_t)));

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
    part.algoState =  state;
}

void bfs_aggregate(Partition<int> &part) {
    bfs_state_t *state = (bfs_state_t *) part.algoState;
    VertexId n = part.vertexCount;
    CUDA_CHECK(D2H(state->levels_h, state->levels, sizeof(int) * n));
    for (VertexId i = 0; i < n; i++) {
        state_g.levels_h[part.globalIds[i]] = state->levels_h[i];
    }
}

template<typename MessageValue>
class Engine {
public:
    /**
     * [config description]
     * @param subgraphs [description]
     */
    void init(const char *path, int numParts) {
        util::enableAllPeerAccess();
        util::expectOverlapOnAllDevices();

        flex::Graph<int, int> graph;
        graph.fromEdgeListFile(path);
        vertexCount = graph.nodes();

        RandomEdgeCut random;
        auto subgraphs = graph.partitionBy(random, numParts);
        partitions.resize(subgraphs.size());
        for (int i = 0; i < subgraphs.size(); i++) {
            partitions[i].fromSubgraph(subgraphs[i]);
        }
    }

    /** allocate local states for each partition */
    void initAlgoStates() {
        double startTime = util::currentTimeMillis();
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            init_bfs(partitions[i]);
            assert(partitions[i].algoState);
        }
        initAlgoStateTime = util::currentTimeMillis() - startTime;
    }

    /** allocate local states for each partition */
    void aggrAlgoStates() {
        double startTime = util::currentTimeMillis();
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            bfs_aggregate(partitions[i]);
        }
        aggrAlgoStateTime = util::currentTimeMillis() - startTime;
    }

    /** Run the engine until all partition vote to quit */
    void run() {
        initAlgoStates();

        supersteps = 0;
        while (true) {
            terminate = true;
            superstep();
            supersteps += 1;

            if (terminate) break;
        }
        aggrAlgoStates();
    }

    void superstep() {
        LOG(DEBUG) << "************************************ Superstep " << supersteps
                   << " ************************************";

        bool *expandLaunched = new bool[partitions.size()];
        bool *scatterLaunched = new bool[partitions.size()];
        bool *compactLaunched = new bool[partitions.size()];
        for (int i = 0; i < partitions.size(); i++) {
            expandLaunched[i] = false;
            scatterLaunched[i] = false;
            compactLaunched[i] = false;
        }

        double startTime = util::currentTimeMillis();
        //////////////////////////// Computation stage /////////////////////////

        if (supersteps >= 1) {

#if 0
            // Peek at inboxes generated in the previous super step
            for (int i = 0; i < partitions.size(); i++) {
                for (int j = i + 1; j < partitions.size(); j++) {
                    printf("p%dinbox%d: ", j, i);
                    partitions[j].inboxes[i].print();
                    printf("p%dinbox%d: ", i, j);
                    partitions[i].inboxes[j].print();
                }
            }
#endif

            // Before launching the kernel, scatter the local state according
            // to the inbox's messages.
            for (int i = 0; i < partitions.size(); i++) {
                for (int j = 0; j < partitions.size(); j++) {
                    if (partitions[i].inboxes[j].length == 0) {
                        continue;
                    }

                    scatterLaunched[i] = true;

                    LOG(DEBUG) << "Partition" << partitions[i].partitionId
                               << " launches a scatter kernel on "
                               << partitions[i].inboxes[j].length << " elements";

                    CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                    CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[0],
                                               partitions[i].streams[1]));
                    auto config = util::kernelConfig(partitions[i].inboxes[j].length);
                    scatterKernel<MessageValue> <<< config.first, config.second, 0,
                                  partitions[i].streams[1]>>>(
                                      &partitions[i].inboxes[j],
                                      partitions[i].algoState,
                                      partitions[i].workset.elemsDevice);
                    CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[0],
                                               partitions[i].streams[1]));
                }
            }

            // Compacting the workset back to the workqueue
            for (int i = 0; i < partitions.size(); i++) {


                compactLaunched[i] = true;

                LOG(DEBUG) << "Partition" << partitions[i].partitionId
                           << " launches a compaction kernel on "
                           << partitions[i].vertexCount << " elements";

                CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                // Clear the queue before generating it
                *partitions[i].workqueueSize = 0;
                CUDA_CHECK(H2D(partitions[i].workqueueSizeDevice,
                               partitions[i].workqueueSize, sizeof(size_t)));

                CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[1],
                                           partitions[i].streams[1]));
                auto config = util::kernelConfig(partitions[i].vertexCount);
                compactKernel <<< config.first, config.second, 0,
                              partitions[i].streams[1]>>>(
                                  partitions[i].workset.elemsDevice,
                                  partitions[i].vertexCount,
                                  partitions[i].workqueue.elemsDevice,
                                  partitions[i].workqueueSizeDevice);
                CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[1],
                                           partitions[i].streams[1]));
            }

            // Transfer all the workqueueSize to host.
            // As long as one partition has work to do, shall not terminate.
            for (int i = 0; i < partitions.size(); i++) {
                CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                CUDA_CHECK(D2H(partitions[i].workqueueSize,
                               partitions[i].workqueueSizeDevice,
                               sizeof(size_t)));
                LOG(DEBUG) << "Partition" << partitions[i].partitionId
                           << " work queue size=" << *partitions[i].workqueueSize;
                if (*partitions[i].workqueueSize != 0) {
                    terminate = false;
                }
            }
        } else {
            terminate = false;
        }

        // return before expansion and message passing starts
        if (terminate == true) return;



        // In each super step, launches the expand kernel for all partitions.
        // The computation kernel is launched in the stream 1.
        // Jump over the partition that has no work to perform.
        for (int i = 0; i < partitions.size(); i++) {

            if (*partitions[i].workqueueSize == 0) {
                continue;
            }

            expandLaunched[i] = true;

            // Clear the outboxes before we put messages to it
            for (int j = 0; j < partitions.size(); j++) {
                if (i == j) continue;
                partitions[i].outboxes[j].clear();
            }

            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches a expansion kernel on "
                       << *partitions[i].workqueueSize << " elements";

            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[2],
                                       partitions[i].streams[1]));
            auto config = util::kernelConfig(*partitions[i].workqueueSize);
            expandKernel<MessageValue> <<< config.first, config.second, 0,
                         partitions[i].streams[1]>>>(
                             partitions[i].partitionId,
                             partitions[i].vertices.elemsDevice,
                             partitions[i].edges.elemsDevice,
                             partitions[i].outboxes,
                             partitions[i].workset.elemsDevice,
                             partitions[i].workqueue.elemsDevice,
                             *partitions[i].workqueueSize,
                             partitions[i].algoState);
            CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[2],
                                       partitions[i].streams[1]));

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
        for (int i = 0; i < partitions.size(); i++) {
            for (int j = 0; j < partitions.size(); j++) {
                if (i == j) continue;
                partitions[i].inboxes[j].swapBuffers();
            }
        }

        //////////////////////////////  Profiling  /////////////////////////////
        // Collect the execution time for each computing kernel.
        // Choose the lagging one to represent the computation time.
        double totalTime = util::currentTimeMillis() - startTime;
        double maxCompTime = 0.0;
        float compTime;
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            float scatterTime = 0.0;
            if (scatterLaunched[i]) {
                CUDA_CHECK(cudaEventElapsedTime(&scatterTime,
                                                partitions[i].startEvents[0],
                                                partitions[i].endEvents[0]));
            }
            float compactTime = 0.0;
            if (compactLaunched[i]) {
                CUDA_CHECK(cudaEventElapsedTime(&compactTime,
                                                partitions[i].startEvents[1],
                                                partitions[i].endEvents[1]));
            }
            float expandTime = 0.0;
            if (expandLaunched[i]) {
                CUDA_CHECK(cudaEventElapsedTime(&expandTime,
                                                partitions[i].startEvents[2],
                                                partitions[i].endEvents[2]));
            }
            compTime = scatterTime + compactTime + expandTime;
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << ": comp=" << std::setprecision(2) << compTime << "ms"
                       << ", scatter=" << std::setprecision(2) << scatterTime / compTime
                       << ", compact=" << std::setprecision(2) << compactTime / compTime
                       << ", expand=" << std::setprecision(2) << expandTime / compTime;

            if (compTime > maxCompTime) maxCompTime = compTime;
        }
        double commTime = totalTime - maxCompTime;
        LOG(INFO) << "Superstep" << supersteps
                  << ": total=" << std::setprecision(3) << totalTime << "ms"
                  << ", comp=" << std::setprecision(2) << maxCompTime / totalTime
                  << ", comm=" << std::setprecision(2) << commTime / totalTime;

        supterstepTime += totalTime;
        supterstepCompTime += maxCompTime;
        supterstepCommTime += commTime;

        delete[] expandLaunched;
        delete[] compactLaunched;
        delete[] scatterLaunched;
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
    inline VertexId getVertexCount() const {
        return vertexCount;
    }

    ~Engine() {
        LOG (INFO) << "Profiling: "
                   << "init=" << std::setprecision(3) << initAlgoStateTime
                   << "ms, aggr=" << std::setprecision(3) << aggrAlgoStateTime
                   << "ms, comp=" << std::setprecision(3) << supterstepCompTime
                   << "ms, comm=" << std::setprecision(3) << supterstepCommTime
                   << "ms, all=" << std::setprecision(3) << supterstepTime
                   << "ms, ";
    }


private:
    int         supersteps;
    bool        terminate;
    VertexId    vertexCount;
    std::vector< Partition<MessageValue> > partitions;
    // For profiling
    double      initAlgoStateTime;
    double      aggrAlgoStateTime;
    double      supterstepCompTime;
    double      supterstepCommTime;
    double      supterstepTime;
};

#endif  // ENGINE_H
