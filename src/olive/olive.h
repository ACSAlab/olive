/**
 * The core components of olive.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-12-20
 * Last Modified: 2014-12-20
 */

#ifndef OLIVE_H
#define OLIVE_H

#include <vector>
#include <iomanip>

#include "common.h"
#include "flexible.h"
#include "partition.h"
#include "partitionStrategy.h"
#include "logging.h"
#include "oliveKernel.h"


template<typename VertexValue>
class Olive {
public:
    /**
     * In each superstep, execute the `edgeContext` for the out-going edges of
     * all active vertices. More specifically, (1) the vertices for which
     * `edgeContex.pred` returns the boolean value true will be filtered out
     * (2) a user-defined function `edgeContex.update` will be applied to them.
     * The filtered-out destination vertices will be marked as active and
     * added to the workset.
     *
     * Since the graph is edge-cutted, some of the destination vertices may be
     * in remote partition, message passing schemes will be used. The whole
     * vertex state will be sent as messages.
     *
     */
    template<typename EdgeContext>
    void edgeFilter(EdgeContext edgeContext) {
        terminate = true;
        // To mask off the cudaEventElapsed API.
        bool *expandLaunched = new bool[partitions.size()];
        bool *scatterLaunched = new bool[partitions.size()];
        bool *compactLaunched = new bool[partitions.size()];
        for (int i = 0; i < partitions.size(); i++) {
            expandLaunched[i] = true;
            scatterLaunched[i] = true;
            compactLaunched[i] = true;
        }

        double startTime = util::currentTimeMillis();
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
        //////////////////////////// Computation stage /////////////////////////
        // Before processing the active vertices, scatter the local state according
        // to the inbox's messages. Skipped if the inbox is empty
        // If there's n partitions, the scatterKernel will be launched n-1 times.
        // So n-1 cudaEvents are required to count the elapsed time.
        for (int i = 0; i < partitions.size(); i++) {
            if (partitions[i].vertexCount == 0) {
                scatterLaunched[i] = false;
                continue;
            }
            for (int j = 0; j < partitions.size(); j++) {
                if (partitions[i].inboxes[j].length == 0) {
                    scatterLaunched[i] = false;
                    continue;
                }

                auto config = util::kernelConfig(partitions[i].inboxes[j].length);
                LOG(DEBUG) << "Partition" << partitions[i].partitionId
                           << " launches a scatter kernel on (" << config.first
                           << "x" << config.second << ")";

                CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[0],
                                           partitions[i].streams[1]));
                {
                    scatterKernel<VertexValue, EdgeContext, true>
                    <<< config.first, config.second, 0, partitions[i].streams[1] >>> (
                        partitions[i].inboxes[j],
                        partitions[i].vertexValues.elemsDevice,
                        partitions[i].workset.elemsDevice,
                        edgeContext);
                }
                CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[0],
                                           partitions[i].streams[1]));
            }
        }

        // Compacting the workset back to the workqueue
        // Skipped if the partition has no vertices.
        for (int i = 0; i < partitions.size(); i++) {
            if (partitions[i].vertexCount == 0) {
                compactLaunched[i] = false;
                continue;
            }

            auto config = util::kernelConfig(partitions[i].vertexCount);
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches a compaction kernel on (" << config.first
                       << "x" << config.second << ")";

            // Clear the queue before generating it
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            *partitions[i].workqueueSize = 0;
            CUDA_CHECK(H2D(partitions[i].workqueueSizeDevice,
                           partitions[i].workqueueSize, sizeof(size_t)));

            CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[1],
                                       partitions[i].streams[1]));
            {
                edgeFilterCompactKernel <<< config.first,
                                        config.second, 0, partitions[i].streams[1]>>> (
                                            partitions[i].workset.elemsDevice,
                                            partitions[i].vertexCount,
                                            partitions[i].workqueue.elemsDevice,
                                            partitions[i].workqueueSizeDevice);
            }
            CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[1],
                                       partitions[i].streams[1]));
        }

        // Transfer all the workqueueSize back to decide whether is terminated.
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

        // In each super step, launches the expand kernel for each partition.
        // The computation kernel is launched in the stream 1.
        // Skipped if the partition has no work to perform (no active vertices).
        for (int i = 0; i < partitions.size(); i++) {
            if (*partitions[i].workqueueSize == 0) {
                expandLaunched[i] = false;
                continue;
            }

            // Clear the outboxes before we put messages to it
            for (int j = 0; j < partitions.size(); j++) {
                if (i == j) continue;
                partitions[i].outboxes[j].clear();
            }

            auto config = util::kernelConfig(*partitions[i].workqueueSize);
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches a expansion kernel on (" << config.first
                       << "x" << config.second << ")";

            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[2],
                                       partitions[i].streams[1]));
            {
                edgeFilterExpandKernel<VertexValue, EdgeContext>
                <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                    partitions[i].partitionId,
                    partitions[i].vertices.elemsDevice,
                    partitions[i].edges.elemsDevice,
                    partitions[i].outboxes,
                    partitions[i].workset.elemsDevice,
                    partitions[i].workqueue.elemsDevice,
                    *partitions[i].workqueueSize,
                    partitions[i].vertexValues.elemsDevice,
                    edgeContext);
            }
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
        LOG(INFO) << "edgeFilter: total=" << std::setprecision(3) << totalTime << "ms"
                  << ", comp=" << std::setprecision(2) << maxCompTime / totalTime
                  << ", comm=" << std::setprecision(2) << commTime / totalTime;

        edgeFilterTime += totalTime;
        edgeFilterCompTime += maxCompTime;
        edgeFilterCommTime += commTime;

        delete[] expandLaunched;
        delete[] compactLaunched;
        delete[] scatterLaunched;

    }

    /**
     * Applies a user-defined function `update` to all vertices in the graph.
     *
     * @param f  Function to update vertex-wise state. It accepts the
     *           original vertex value as parameter and returns a new
     *           vertex value.
     */
    template<typename VertexFunction>
    void vertexMap(VertexFunction f) {
        for (int i = 0; i < partitions.size(); i++) {
            auto config = util::kernelConfig(partitions[i].vertexValues.size());
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches a vertexMap kernel (" << config.first
                       << "x" << config.second << ")";

            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            {
                vertexMapKernel <VertexFunction, VertexValue>
                <<< config.first, config.second>>>(
                    partitions[i].vertexValues.elemsDevice,
                    partitions[i].vertexValues.size(),
                    f);
            }
            CUDA_CHECK(cudaThreadSynchronize());
        }
    }

    /**
     * Filters one vertex by specifying a vertex id and applies a user-defined
     * function `update` to it. The filtered-out vertex will be marked as active
     * and added to the workset.
     *
     * Concerns are that if an algorithm requires filtering a bunch of vertices,
     * the the kernel is invoked frequently. e.g. in Radii Estimation.
     *
     * @param id      Takes the vertex id to filter as parameter
     * @param f       Function to update vertex-wise state. It accepts the
     *                original vertex value as parameter and returns a new
     *                vertex value.
     */
    template<typename VertexFunction>
    void vertexFilter(VertexId id, VertexFunction f) {
        for (int i = 0; i < partitions.size(); i++) {
            auto config = util::kernelConfig(partitions[i].vertexValues.size());
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches a vertexFilter kernel (" << config.first
                       << "x" << config.second << ")";

            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            {
                vertexFilterKernel <VertexFunction, VertexValue>
                <<< config.first, config.second>>> (
                    partitions[i].globalIds.elemsDevice,
                    partitions[i].vertexValues.size(),
                    id,
                    partitions[i].vertexValues.elemsDevice,
                    f,
                    partitions[i].workset.elemsDevice);
            }
            CUDA_CHECK(cudaThreadSynchronize());
        }

        terminate = false;
    }

    /**
     * Iterate over all local vertex states, and applies a UDF to them. The UDF
     * knows the global index to put the vertex.
     *
     * @param f     Function to update global states. It accept the offset in
     *              global buffers as the 1st parameter and the local vertex
     *              value as the 2nd parameter.
     */
    template<typename VertexAtFunction>
    void vertexTransform(VertexAtFunction f) {
        for (int i = 0; i < partitions.size(); i++) {
            partitions[i].vertexValues.persist();
            for (VertexId j = 0; j < partitions[i].vertexValues.size(); j++) {
                f(partitions[i].globalIds[j], partitions[i].vertexValues[j]);
            }
        }
    }

    /**
     * Initialize the engine by specifying a graph path and the number of
     * partitions. (random partition by default)
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

        terminate = true;
    }

    /** Returns the number of the vertices in the graph. */
    inline VertexId getVertexCount() const {
        return vertexCount;
    }

    /** Returns the termination state of the engine. */
    inline bool isTerminated() const {
        return terminate;
    }

    /** Log the profile in deconstructor */
    ~Olive() {
        LOG (INFO) << "edgeFilter: "
                   << "ms, comp=" << std::setprecision(3) << edgeFilterCompTime
                   << "ms, comm=" << std::setprecision(3) << edgeFilterCommTime
                   << "ms, all=" << std::setprecision(3) << edgeFilterTime
                   << "ms, ";
    }

private:
    bool        terminate;
    VertexId    vertexCount;

    /**
     * For each partition the whole state of vertex will be treated as message
     */
    std::vector< Partition<VertexValue, VertexValue> > partitions;

    /** For profiling */
    double      edgeFilterCompTime;
    double      edgeFilterCommTime;
    double      edgeFilterTime;
};

#endif  // OLIVE_H
