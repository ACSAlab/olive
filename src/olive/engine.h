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


/**
 * The scatter kernel operates on each inbox and updates the local values.
 */
template<typename VertexValue, typename MessageValue>
__global__
void scatterKernel(
    MessageBox< VertexMessage<MessageValue> > *inbox,
    VertexValue *vertexValues,
    int *workset,
    bool (*cond)(VertexValue),
    VertexValue (*update)(VertexValue),
    VertexValue (*unpack)(MessageValue))
{
    int tid = THREAD_INDEX;
    if (tid >= inbox->length) return;

    VertexId inNode = inbox->buffer[tid].receiverId;
    VertexValue newValue = unpack(inbox->buffer[tid].value);

    if (cond(vertexValues[inNode])) {
        vertexValues[inNode] = update(newValue);
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

template<typename VertexValue, typename MessageValue>
__global__
void expandKernel(
    PartitionId thisPid,
    const EdgeId *vertices,
    const Vertex *edges,
    MessageBox< VertexMessage<MessageValue> > *outboxes,
    int *workset,
    VertexId *workqueue,
    int n,
    VertexValue *vertexValues,
    bool (*cond)(VertexValue),
    VertexValue (*update)(VertexValue),
    MessageValue (*pack)(VertexValue))
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
            if (cond(vertexValues[inNode])) {
                vertexValues[inNode] = update(vertexValues[outNode]);
                workset[inNode] = 1;
            }
        } else {  // In remote partition
            VertexMessage<MessageValue> msg;
            msg.receiverId = edges[edge].localId;
            msg.value = pack(vertexValues[outNode]);

            size_t offset = atomicAdd(reinterpret_cast<unsigned long long *> (&outboxes[pid].length), 1);
            outboxes[pid].buffer[offset] = msg;
        }
    }
}

template<typename VertexValue>
__global__
void vertexMapKernel(
    VertexValue *vertexValues,
    int n,
    VertexValue (*f)(VertexValue))
{
    int tid = THREAD_INDEX;
    if (tid >= n) return;
    vertexValues[tid] = f(vertexValues[tid]);
}


template<typename VertexValue>
__global__
void vertexFilterKernel(
    const VertexId *globalIds,
    int n,
    VertexId id,
    VertexValue *vertexValues,
    VertexValue (*f)(VertexValue),
    int *workset)
{
    int tid = THREAD_INDEX;
    if (tid >= n) return;
    if (globalIds[tid] == id) {
        vertexValues[tid] = f(vertexValues[tid]);
        workset[tid] = 1;
    }
}


template<typename VertexValue, typename MessageValue>
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

    /**
     * Map a UDF to the global states to aggregate results.
     * Mapping the global id from partition local id via `globalIds`.
     */
    void aggregate(void (*f)(VertexId, VertexValue)) {
        double startTime = util::currentTimeMillis();
        for (int i = 0; i < partitions.size(); i++) {
            partitions[i].vertexValues.persist();
            for (VertexId j = 0; j < partitions[i].vertexValues.size(); j++) {
                f(partitions[i].globalIds[j],
                partitions[i].vertexValues[j]);
            }
        }
        LOG(INFO) << "It took " << std::setprecision(3)
                  << util::currentTimeMillis() - startTime
                  << "ms to aggregate results.";
    }


    typedef VertexValue (*VertexFunctor) (VertexValue);

    /**
     * Maps a UDF `update` to each vertex.
     */
    void vertexMap(VertexValue (*update) (VertexValue)) {

        for (int i = 0; i < partitions.size(); i++) {

            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches a vertexMap kernel on "
                       << partitions[i].vertexValues.size() << " elements";

            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            auto config = util::kernelConfig(partitions[i].vertexValues.size());
            vertexMapKernel <VertexValue> <<< config.first, config.second>>>(
                partitions[i].vertexValues.elemsDevice,
                partitions[i].vertexValues.size(),
                update);
            CUDA_CHECK(cudaThreadSynchronize());
        }
    }

    /**
     * Maps a UDF `update` to the vertex with global `id`.
     * The filtered out vertex will be added to the workset.
     */
    void vertexFilter(VertexId id, VertexValue (*update)(VertexValue)) {
        for (int i = 0; i < partitions.size(); i++) {

            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches a vertexFilter kernel on "
                       << partitions[i].vertexValues.size() << " elements";

            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            auto config = util::kernelConfig(partitions[i].vertexValues.size());
            vertexFilterKernel <VertexValue> <<< config.first, config.second>>>(
                partitions[i].globalIds.elemsDevice,
                partitions[i].vertexValues.size(),
                id,
                partitions[i].vertexValues.elemsDevice,
                update,
                partitions[i].workset.elemsDevice);
            CUDA_CHECK(cudaThreadSynchronize());
        }
    }

    /** Run the engine until all partition vote to quit */
    void run(bool (*cond)(VertexValue),
             VertexValue (*update)(VertexValue),
             MessageValue (*pack)(VertexValue),
             VertexValue (*unpack)(MessageValue))
    {
        supersteps = 0;
        while (true) {
            terminate = true;
            superstep(cond, update, pack, unpack);

            if (terminate) break;
        }
    }

    void superstep(bool (*cond)(VertexValue),
             VertexValue (*update)(VertexValue),
             MessageValue (*pack)(VertexValue),
             VertexValue (*unpack)(MessageValue)) 
    {
        LOG(DEBUG) << "************************************ Superstep " << supersteps
                   << " ************************************";

        // To mask off the cudaEventElapsed API.
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
                scatterKernel<VertexValue, MessageValue> <<< config.first, config.second, 0,
                              partitions[i].streams[1] >>> (
                                  &partitions[i].inboxes[j],
                                  partitions[i].vertexValues.elemsDevice,
                                  partitions[i].workset.elemsDevice,
                                  cond,
                                  update,
                                  unpack);
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

        // Returns before expansion and message passing starts
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
            expandKernel<VertexValue, MessageValue> <<< config.first, config.second, 0,
                         partitions[i].streams[1]>>>(
                             partitions[i].partitionId,
                             partitions[i].vertices.elemsDevice,
                             partitions[i].edges.elemsDevice,
                             partitions[i].outboxes,
                             partitions[i].workset.elemsDevice,
                             partitions[i].workqueue.elemsDevice,
                             *partitions[i].workqueueSize,
                             partitions[i].vertexValues.elemsDevice,
                             cond,
                             update,
                             pack);
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

        supersteps += 1;
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
                   << "ms, comp=" << std::setprecision(3) << supterstepCompTime
                   << "ms, comm=" << std::setprecision(3) << supterstepCommTime
                   << "ms, all=" << std::setprecision(3) << supterstepTime
                   << "ms, ";
    }


private:
    int         supersteps;
    bool        terminate;
    VertexId    vertexCount;
    std::vector< Partition<VertexValue, MessageValue> > partitions;
    // For profiling
    double      supterstepCompTime;
    double      supterstepCommTime;
    double      supterstepTime;
};

#endif  // ENGINE_H
