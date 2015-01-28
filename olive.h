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
#include "timer.h"
#include "utils.h"
#include "commandLine.h"
#include "oliveKernel.h"


template<typename VertexValue, typename AccumValue>
class Olive {
public:
    /**
     *
     * Since the graph is edge-cutted, some of the destination vertices may be
     * in remote partition, message passing schemes will be used. The whole
     * vertex state will be sent as messages.
     *
     */
    template<typename F>
    void edgeMap(F f) {
        double startTime = getTimeMillis();

        //////////////////////////// Computation stage /////////////////////////
        // In each super step, launches the edgeMap kernel for each partition.
        // The computation kernel is launched in the stream 1.
        // Skipped if the partition has no work to perform (no active vertices).
        for (int i = 0; i < partitions.size(); i++) {
            assert(partitions[i].vertexCount > 0);
            // Clear the accumulator before the gather phase starts
            partitions[i].accumulators.allTo(0);
            // Clear the outboxes before we put messages to it
            for (int rmtPid = 0; rmtPid < partitions.size(); rmtPid++) {
                if (rmtPid == i) continue;
                partitions[i].outboxes[rmtPid].clear();
            }
            // Transfer the queue size back
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(D2H(partitions[i].workqueueSize,
                           partitions[i].workqueueSizeDevice,
                           sizeof(VertexId)));

            LOG(DEBUG) << "Partition " << partitions[i].partitionId
                       << " work queue size=" << *partitions[i].workqueueSize;

            auto config = util::kernelConfig(*partitions[i].workqueueSize);
            CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[0], partitions[i].streams[1]));
            {
                edgeGatherKernel<VertexValue, AccumValue, F>
                <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                    partitions[i].partitionId,
                    partitions[i].workqueue.elemsDevice,
                    partitions[i].workqueueSizeDevice,
                    partitions[i].vertices.elemsDevice,
                    partitions[i].edges.elemsDevice,
                    partitions[i].vertexValues.elemsDevice,
                    partitions[i].accumulators.elemsDevice,
                    partitions[i].workset.elemsDevice,
                    partitions[i].outboxes,
                    f);
            }
            CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[0], partitions[i].streams[1]));
        }

        ///////////////////////// Communication stage //////////////////////////
        // All-to-all message box transferring.
        //
        // To satisfy the dependency, the asynchronous copy can only be launched
        // in the same stream as that of the source partition.
        //
        // The copy will be launched strictly after the source partition has
        // finished the gather phase on the local vertices, and got all outboxes
        // eady.
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
            CUDA_CHECK(cudaStreamSynchronize(partitions[i].streams[1]));
        }

        //////////////////////////////  Profiling  /////////////////////////////
        // Collect the execution time for each computing kernel.
        // Choose the lagging one to represent the computation time.
        double totalTime = getTimeMillis() - startTime;
        double maxCompTime = 0.0;
        for (int i = 0; i < partitions.size(); i++) {
            float compTime;
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventElapsedTime(&compTime, partitions[i].startEvents[0],
                                            partitions[i].endEvents[0]));

            if (compTime > maxCompTime) maxCompTime = compTime;
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << ": edgeMapGather=" << std::setprecision(2) << compTime
                       << "ms";
        }
        double commTime = totalTime - maxCompTime;
        LOG(INFO) << "edgeMapGather: total=" << std::setprecision(3) << totalTime
                  << "ms, comp=" << std::setprecision(2) << maxCompTime
                  << "ms, comm=" << std::setprecision(2) << commTime << "ms";


        // Scatter the local state according to the inbox's messages.
        // Launch the scatter kernel for each inbox
        // Skipped if the inbox is empty
        startTime = getTimeMillis();
        for (int i = 0; i < partitions.size(); i++) {
            for (int rmtPid = 0; rmtPid < partitions.size(); rmtPid++) {
                if (rmtPid == i) continue;
                if (partitions[i].inboxes[rmtPid].length == 0) continue;
                auto config = util::kernelConfig(partitions[i].inboxes[rmtPid].length);
                CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                {
                    edgeScatterKernel<AccumValue, F>
                    <<< config.first, config.second, 0, partitions[i].streams[1] >>>(
                        partitions[i].inboxes[rmtPid],
                        partitions[i].accumulators.elemsDevice,
                        partitions[i].workset.elemsDevice,
                        f);
                }
                LOG(DEBUG) << "Partition" << partitions[i].partitionId
                           << " from " << rmtPid << " message size="
                           << partitions[i].inboxes[rmtPid].length;
            }
        }
        // Synchronize all partitions
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaStreamSynchronize(partitions[i].streams[1]));
        }

        for (int i = 0; i < partitions.size(); i++) {
            for (int j = 0; j < partitions.size(); j++) {
                if (i == j) continue;
                partitions[i].inboxes[j].swapBuffers();
            }
        }

        // Profiling the time
        totalTime = getTimeMillis() - startTime;
        LOG(INFO) << "edgeMapScatter=" << std::setprecision(2) << totalTime << "ms";


        // Peek the activated vertices in vertex phase
        for (int i = 0; i < partitions.size(); i++) {
            partitions[i].workset.persist();
            for (int j = 0; j < partitions[i].workset.length; j++) {
                printf("%d: %d\n", partitions[i].globalIds[j], partitions[i].workset[j]);
            }
        }
    }

    /**
     * Applies a user-defined function `update` to all vertices in the graph.
     *
     * @param f  Function to update vertex-wise state. It accepts the
     *           original vertex value as parameter and returns a new
     *           vertex value.
     */
    // template<typename F>
    // void vertexMap(F f) {
    //     double startTime = getTimeMillis();
    //     // Synchronize all partitions
    //     for (auto &par : partitions) {
    //         assert(par.vertexCount > 0);
    //         auto config = util::kernelConfig(par.vertexCount);
    //         CUDA_CHECK(cudaSetDevice(par.deviceId));
    //         CUDA_CHECK(cudaEventRecord(par.startEvents[0], par.streams[1]));
    //         {
    //             vertexMapKernel<VertexValue, AccumValue, F>
    //             <<< config.first, config.second, 0, par.streams[1]>>>(
    //                 par.vertexCount,
    //                 par.vertexValues.elemsDevice,
    //                 par.accumulators.elemsDevice,
    //                 f);
    //         }
    //         CUDA_CHECK(cudaEventRecord(par.endEvents[0], par.streams[1]));
    //     }
    //     // Synchronize all partitions
    //     for (auto &par : partitions) {
    //         CUDA_CHECK(cudaStreamSynchronize(par.streams[1]));
    //     }
    //     // Profiling the time
    //     double totalTime = getTimeMillis() - startTime;
    //     for (auto &par : partitions) {
    //         float time;
    //         CUDA_CHECK(cudaSetDevice(par.deviceId));
    //         CUDA_CHECK(cudaEventElapsedTime(&time, par.startEvents[0],
    //                                         par.endEvents[0]));
    //         LOG(DEBUG) << "Partition" << par.partitionId << " vertexMap: "
    //                    << " vertex count=" << par.vertexCount
    //                    << " time="  << std::setprecision(2) << time << "ms";
    //     }
    //     LOG(INFO) << "vertexMap=" << std::setprecision(2) << totalTime << "ms";
    // }

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
    template<typename F>
    void vertexMap(F f) {
        double startTime = getTimeMillis();
        for (int i = 0; i < partitions.size(); i++) {
            assert(partitions[i].vertexCount > 0);
            // Clear the workqueue before generating it
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            *partitions[i].workqueueSize = 0;
            CUDA_CHECK(H2D(partitions[i].workqueueSizeDevice,
                           partitions[i].workqueueSize, sizeof(VertexId)));

            auto config = util::kernelConfig(partitions[i].vertexCount);
            CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[0], partitions[i].streams[1]));
            {
                vertexMapKernel<VertexValue, AccumValue, F>
                <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                    partitions[i].workset.elemsDevice,
                    partitions[i].vertexCount,
                    partitions[i].vertexValues.elemsDevice,
                    partitions[i].accumulators.elemsDevice,
                    partitions[i].workqueue.elemsDevice,
                    partitions[i].workqueueSizeDevice,
                    f);
            }
            CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[0], partitions[i].streams[1]));
        }
        // Synchronize all partitions
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaStreamSynchronize(partitions[i].streams[1]));
        }
        // Profiling the time
        double totalTime = getTimeMillis() - startTime;
        for (int i = 0; i < partitions.size(); i++) {
            float time;
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventElapsedTime(&time, partitions[i].startEvents[0],
                                            partitions[i].endEvents[0]));
            LOG(DEBUG) << "Partition" << partitions[i].partitionId << " vertexMap: "
                       << " vertex count=" << partitions[i].vertexCount
                       << " time="  << std::setprecision(2) << time << "ms";
        }
        LOG(INFO) << "vertexMap=" << std::setprecision(2) << totalTime << "ms";

        // Peek the activated vertices in edge phase
        for (int i = 0; i < partitions.size(); i++) {
            partitions[i].workqueue.persist();
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(D2H(partitions[i].workqueueSize,
                           partitions[i].workqueueSizeDevice,
                           sizeof(VertexId)));
            for (int j = 0; j < *partitions[i].workqueueSize; j++) {
                printf("%d ", partitions[i].workqueue[j]);
            }
            printf("\n");
        }

    }


    template<typename F>
    void vertexFilter(F f) {
        double startTime = getTimeMillis();
        for (int i = 0; i < partitions.size(); i++) {
            assert(partitions[i].vertexCount > 0);
            // Clear the workqueue before generating it
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            *partitions[i].workqueueSize = 0;
            CUDA_CHECK(H2D(partitions[i].workqueueSizeDevice,
                           partitions[i].workqueueSize, sizeof(VertexId)));

            auto config = util::kernelConfig(partitions[i].vertexCount);
            CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[0], partitions[i].streams[1]));
            {
                vertexFilterKernel<VertexValue, AccumValue, F>
                <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                    partitions[i].workset.elemsDevice,
                    partitions[i].vertexCount,
                    partitions[i].vertexValues.elemsDevice,
                    partitions[i].workqueue.elemsDevice,
                    partitions[i].workqueueSizeDevice,
                    f);
            }
            CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[0], partitions[i].streams[1]));
        }
        // Synchronize all partitions
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaStreamSynchronize(partitions[i].streams[1]));
        }
        // Profiling the time
        double totalTime = getTimeMillis() - startTime;
        for (int i = 0; i < partitions.size(); i++) {
            float time;
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventElapsedTime(&time, partitions[i].startEvents[0],
                                            partitions[i].endEvents[0]));
            LOG(DEBUG) << "Partition" << partitions[i].partitionId << " vertexFilter: "
                       << " vertex count=" << partitions[i].vertexCount
                       << " time="  << std::setprecision(2) << time << "ms";
        }
        LOG(INFO) << "vertexFilter=" << std::setprecision(2) << totalTime << "ms";
    }


    /**
     * Transfer all the `workqueueSize` back and sum them up.
     */
    inline VertexId getWorkqueueSize() {
        VertexId size = 0;
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(D2H(partitions[i].workqueueSize,
                           partitions[i].workqueueSizeDevice,
                           sizeof(VertexId)));
            // LOG(DEBUG) << "Partition " << partitions[i].partitionId
            //            << " work queue size=" << *partitions[i].workqueueSize;
            size += *partitions[i].workqueueSize;
        }
        return size;
    }

    /**
     * Clear the work queue.
     */
    inline void clearWorkqueue() {
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            *partitions[i].workqueueSize = 0;
            CUDA_CHECK(H2D(partitions[i].workqueueSizeDevice,
                           partitions[i].workqueueSize, sizeof(VertexId)));
        }
    }

    /**
     * Transfer all the `allVerticesInactive` back and sum them up.
     */
    inline bool allVerticesInactive() {
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(D2H(partitions[i].allVerticesInactive,
                           partitions[i].allVerticesInactiveDevice,
                           sizeof(bool)));
            if (!*partitions[i].allVerticesInactive) {
                return false;
            }
        }
        return true;
    }


    /**
     * Iterate over all local vertex states, and applies a UDF to them. The UDF
     * knows the global index to put the vertex.
     *
     * @param f     Function to update global states. It accept the offset in
     *              global buffers as the 1st parameter and the local vertex
     *              value as the 2nd parameter.
     */
    template<typename F>
    void vertexTransform(F f) {
        for (int i = 0; i < partitions.size(); i++) {
            partitions[i].vertexValues.persist();

            for (VertexId j = 0; j < partitions[i].vertexValues.size(); j++) {
                f(partitions[i].globalIds[j],
                  partitions[i].vertexValues[j]);
            }
        }
    }

    /**
     * Initialize the engine by specifying a graph path and the number of
     * partitions. (random partition by default)
     */
    void readGraph(const char *path, int numParts) {
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

    /** Returns the number of the vertices in the graph. */
    inline VertexId getVertexCount() const {
        return vertexCount;
    }


private:
    VertexId    vertexCount;

    /**
     * For each partition the whole state of vertex will be treated as message
     */
    std::vector< Partition<VertexValue, AccumValue> > partitions;
};

#endif  // OLIVE_H
