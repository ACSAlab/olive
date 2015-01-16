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
    template<typename EdgeFunction>
    void edgeMap(EdgeFunction f) {
        // To mask off the cudaEventElapsed API.
        bool *gatherLaunched = new bool[partitions.size()];
        bool *scatterLaunched = new bool[partitions.size()];
        for (int i = 0; i < partitions.size(); i++) {
            gatherLaunched[i] = true;
            scatterLaunched[i] = true;
        }
        double startTime = util::currentTimeMillis();

        //  Clear the accumulator before the computation starts
        for (int i = 0; i < partitions.size(); i++) {
            partitions[i].accumulators.allTo(0);
        }

        //////////////////////////// Computation stage /////////////////////////
        // In each super step, launches the edgeMap kernel for each partition.
        // The computation kernel is launched in the stream 1.
        // Skipped if the partition has no work to perform (no active vertices).
        for (int i = 0; i < partitions.size(); i++) {
            // if (*partitions[i].workqueueSize == 0) {
            //     gatherLaunched[i] = false;
            //     continue;
            // }

            // Clear the outboxes before we put messages to it
            for (int j = 0; j < partitions.size(); j++) {
                if (i == j) continue;
                partitions[i].outboxes[j].clear();
            }

            auto config = util::kernelConfig(partitions[i].vertexCount);
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " vertex count=" << partitions[i].vertexCount;
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches edgeGatherDense kernel (" << config.first
                       << "x" << config.second << ")";

            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[1],
                                       partitions[i].streams[1]));
            {
                edgeGatherDenseKernel<VertexValue, AccumValue, EdgeFunction>
                <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                    partitions[i].partitionId,
                    partitions[i].vertices.elemsDevice,
                    partitions[i].edges.elemsDevice,
                    partitions[i].outboxes,
                    partitions[i].vertexCount,
                    partitions[i].vertexValues.elemsDevice,
                    partitions[i].accumulators.elemsDevice,
                    f);
            }
            CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[1],
                                       partitions[i].streams[1]));

            // LOG(DEBUG) << "Partition" << partitions[i].partitionId
            //            << " work queue size=" << *partitions[i].workqueueSize;
            // auto config = util::kernelConfig(*partitions[i].workqueueSize);
            // LOG(DEBUG) << "Partition" << partitions[i].partitionId
            //            << " launches edgeGather kernel (" << config.first
            //            << "x" << config.second << ")";
            // CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            // CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[1],
            //                            partitions[i].streams[1]));
            // {
            //     edgeGatherDenseKernel<VertexValue, AccumValue, EdgeFunction>
            //     <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
            //         partitions[i].partitionId,
            //         partitions[i].vertices.elemsDevice,
            //         partitions[i].edges.elemsDevice,
            //         partitions[i].outboxes,
            //         partitions[i].workqueue.elemsDevice,
            //         *partitions[i].workqueueSize,
            //         partitions[i].vertexValues.elemsDevice,
            //         partitions[i].accumulators.elemsDevice,
            //         f);
            // }
            // CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[1],
            //                            partitions[i].streams[1]));


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


        // scatter the local state according
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
                           << " inbox" << j
                           << " size=" << partitions[i].inboxes[j].length;
                LOG(DEBUG) << "Partition" << partitions[i].partitionId
                           << " launches edgeScatter kernel (" << config.first
                           << "x" << config.second << ")";

                CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[0],
                                           partitions[i].streams[1]));
                {
                    edgeScatterKernel<AccumValue, EdgeFunction>
                    <<< config.first, config.second, 0, partitions[i].streams[1] >>> (
                        partitions[i].inboxes[j],
                        partitions[i].accumulators.elemsDevice,
                        f);
                }
                CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[0],
                                           partitions[i].streams[1]));
            }
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
        for (int i = 0; i < partitions.size(); i++) {
            float compTime;
            float scatterTime = 0.0;
            float gatherTime = 0.0;
            if (scatterLaunched[i] || gatherLaunched[i]) {
                CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            }
            if (scatterLaunched[i]) {
                CUDA_CHECK(cudaEventElapsedTime(&scatterTime,
                                                partitions[i].startEvents[0],
                                                partitions[i].endEvents[0]));
            }
            if (gatherLaunched[i]) {
                CUDA_CHECK(cudaEventElapsedTime(&gatherTime,
                                                partitions[i].startEvents[1],
                                                partitions[i].endEvents[1]));
            }
            compTime = scatterTime + gatherTime;
            if (compTime > maxCompTime) maxCompTime = compTime;
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << ": edgeMap: total=" << std::setprecision(2) << compTime
                       << "ms, scatter=" << std::setprecision(2) << scatterTime
                       << "ms, gather="  << std::setprecision(2) << gatherTime
                       << "ms";
        }
        double commTime = totalTime - maxCompTime;
        LOG(INFO) << "edgeMap: total=" << std::setprecision(3) << totalTime
                  << "ms, comp=" << std::setprecision(2) << maxCompTime
                  << "ms, comm=" << std::setprecision(2) << commTime
                  << "ms";

        oliveTotalTime += totalTime;
        oliveCompTime += maxCompTime;
        oliveCommTime += commTime;

        delete[] gatherLaunched;
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

        bool *kernelLaunched = new bool[partitions.size()];
        for (int i = 0; i < partitions.size(); i++) {
            kernelLaunched[i] = true;
        }

        for (int i = 0; i < partitions.size(); i++) {
            if (partitions[i].vertexCount == 0) {
                kernelLaunched[i] = false;
                continue;
            }

            auto config = util::kernelConfig(partitions[i].vertexCount);
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " vertex count=" << partitions[i].vertexCount;
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches vertexMap kernel (" << config.first
                       << "x" << config.second << ")";

            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[0],
                                       partitions[i].streams[1]));
            {
                vertexMapKernel<VertexValue, AccumValue, VertexFunction>
                <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                    partitions[i].vertexValues.elemsDevice,
                    partitions[i].vertexCount,
                    partitions[i].accumulators.elemsDevice,                    
                    f);
            }
            CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[0],
                                       partitions[i].streams[1]));
            CUDA_CHECK(cudaStreamSynchronize(partitions[i].streams[1]));
        }

        // Profiling the time
        double maxTime = 0.0;
        for (int i = 0; i < partitions.size(); i++) {
            float time;
            if (kernelLaunched[i]) {
                CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                CUDA_CHECK(cudaEventElapsedTime(&time,
                                                partitions[i].startEvents[0],
                                                partitions[i].endEvents[0]));
            }
            if (time > maxTime) maxTime = time;
        }
        LOG(INFO)  <<"vertexMap: "<< std::setprecision(2) << maxTime << "ms";

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
    void vertexFilter(VertexFunction f) {

        bool *kernelLaunched = new bool[partitions.size()];
        for (int i = 0; i < partitions.size(); i++) {
            kernelLaunched[i] = true;
        }

        for (int i = 0; i < partitions.size(); i++) {
            if (partitions[i].vertexCount == 0) {
                kernelLaunched[i] = false;
                continue;
            }

            auto config = util::kernelConfig(partitions[i].vertexCount);
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " vertex count=" << partitions[i].vertexCount;
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " launches vertexFilter kernel (" << config.first
                       << "x" << config.second << ")";

            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(cudaEventRecord(partitions[i].startEvents[0],
                                       partitions[i].streams[1]));
            {
                vertexFilterKernel<VertexValue, AccumValue, VertexFunction>
                <<< config.first, config.second, 0, partitions[i].streams[1]>>>(
                    partitions[i].workset.elemsDevice,
                    partitions[i].vertexCount,
                    partitions[i].workqueue.elemsDevice,
                    partitions[i].workqueueSizeDevice,
                    partitions[i].vertexValues.elemsDevice,
                    partitions[i].accumulators.elemsDevice,
                    f);
            }
            CUDA_CHECK(cudaEventRecord(partitions[i].endEvents[0],
                                       partitions[i].streams[1]));
            CUDA_CHECK(cudaStreamSynchronize(partitions[i].streams[1]));
        }

        // Profiling the time
        double maxTime = 0.0;
        for (int i = 0; i < partitions.size(); i++) {
            float time;
            if (kernelLaunched[i]) {
                CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
                CUDA_CHECK(cudaEventElapsedTime(&time,
                                                partitions[i].startEvents[0],
                                                partitions[i].endEvents[0]));
            }
            if (time > maxTime) maxTime = time;
        }
        LOG(INFO)  <<"vertexFilter: "<< std::setprecision(2) << maxTime << "ms";
    }


    /**
     * Transfer all the workqueueSize back to decide whether is terminated.
     * As long as one partition has work to do, shall not terminate.
     */
    inline VertexId getWorksetSize() {
        VertexId size;
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            CUDA_CHECK(D2H(partitions[i].workqueueSize,
                           partitions[i].workqueueSizeDevice,
                           sizeof(VertexId)));
            LOG(DEBUG) << "Partition" << partitions[i].partitionId
                       << " work queue size=" << *partitions[i].workqueueSize;
            size += *partitions[i].workqueueSize;
        }
        return size;
    }

    /**
     * Clear the work queue
     */
    inline void clearWorkset() {
        for (int i = 0; i < partitions.size(); i++) {
            CUDA_CHECK(cudaSetDevice(partitions[i].deviceId));
            *partitions[i].workqueueSize = 0;
            CUDA_CHECK(H2D(partitions[i].workqueueSizeDevice,
                           partitions[i].workqueueSize, sizeof(VertexId)));
        }
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
    }

    /** Returns the number of the vertices in the graph. */
    inline VertexId getVertexCount() const {
        return vertexCount;
    }

    /** Log the profile in deconstructor */
    ~Olive() {
        LOG (INFO) << "Olive: all=" << std::setprecision(3) << oliveTotalTime
                   << "ms, comp=" << std::setprecision(3) << oliveCompTime
                   << "ms, comm=" << std::setprecision(3) << oliveCommTime
                   << "ms, ";
    }

private:
    VertexId    vertexCount;

    /**
     * For each partition the whole state of vertex will be treated as message
     */
    std::vector< Partition<VertexValue, AccumValue> > partitions;

    /** For profiling */
    double      oliveTotalTime;
    double      oliveCompTime;
    double      oliveCommTime;
};

#endif  // OLIVE_H
