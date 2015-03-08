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
 * The single GPU version of Olive
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2015-02-05
 * Last Modified: 2015-02-05
 */

#ifndef OLIVER_H
#define OLIVER_H

#include "common.h"
#include "csrGraph.h"
#include "logging.h"
#include "timer.h"
#include "utils.h"
#include "commandLine.h"
#include "grd.h"
#include "vertexSubset.h"
#include "oliverKernel.h"

template<typename VertexValue, typename AccumValue>
class Oliver {
public:
    /**
     * The edgeMap function accepts a dense vertexSubset and produces
     * a sparse one.
     */
    template<typename F>
    void edgeMap(VertexSubset dstV, VertexSubset srcV, F f) {

        // Clear the accumulator before the gather phase starts
        accumulators.allTo(0);

        auto c = util::kernelConfig(srcV.size());
        edgeMapKernel<VertexValue, AccumValue, F> <<< c.first, c.second>>>(
            srcV.workqueue.elemsDevice,
            srcV.qSizeDevice,
            srcVertices.elemsDevice,
            outgoingEdges.elemsDevice,
            vertexValues.elemsDevice,
            accumulators.elemsDevice,
            dstV.workset.elemsDevice,
            f);
        CUDA_CHECK(cudaThreadSynchronize());
    }

    /**
     * The vertexFilter function accepts a sparse vertexSubset and produces
     * a dense one.
     */
    template<typename F>
    void vertexFilter(VertexSubset dstV, VertexSubset srcV, F f) {
        auto c = util::kernelConfig(vertexCount);
        vertexFilterKernel<VertexValue, AccumValue, F> <<< c.first, c.second>>>(
            srcV.workset.elemsDevice,
            vertexCount,
            vertexValues.elemsDevice,
            accumulators.elemsDevice,
            dstV.workqueue.elemsDevice,
            dstV.qSizeDevice,
            f);
        CUDA_CHECK(cudaThreadSynchronize());
    }

    /**
     * vertexMap is used to update the local vertex state. 
     * The computation is topology-independent.
     */
    template<typename F>
    void vertexMap(VertexSubset srcV, F f) {
        auto c = util::kernelConfig(vertexCount);
        vertexMapKernel<VertexValue, AccumValue, F> <<< c.first, c.second>>>(
            srcV.workset.elemsDevice,
            vertexCount,
            vertexValues.elemsDevice,
            f);
        CUDA_CHECK(cudaThreadSynchronize());
    }

    void readGraph(const CsrGraph<int, int> &graph) {
        vertexCount = graph.vertexCount;
        edgeCount = graph.edgeCount;
        srcVertices.reserve(vertexCount + 1);
        outgoingEdges.reserve(edgeCount);
        vertexValues.reserve(vertexCount);
        accumulators.reserve(vertexCount);
        memcpy(srcVertices.elemsHost, graph.vertices, sizeof(EdgeId) * (vertexCount + 1));
        memcpy(outgoingEdges.elemsHost, graph.edges, sizeof(VertexId) * edgeCount);
        srcVertices.cache();
        outgoingEdges.cache();
    }

    inline void printVertices() {
        vertexValues.persist();
        vertexValues.print();
    }

    /** Returns the number of the vertices in the graph. */
    inline VertexId getVertexCount() const {
        return vertexCount;
    }

    ~Oliver() {
        srcVertices.del();
        outgoingEdges.del();
        vertexValues.del();
        accumulators.del();
    }

private:
    /** Record the edge and vertex number of each partition. */
    VertexId         vertexCount;
    EdgeId           edgeCount;

    /**
     * CSR related data structure.
     */
    GRD<EdgeId>      srcVertices;
    GRD<VertexId>    outgoingEdges;

    // GRD<EdgeId>      dstVertices;
    // GRD<VertexId>    incomingEdges;

    /**
     * Vertex-wise state.
     */
    GRD<VertexValue> vertexValues;
    GRD<AccumValue>  accumulators;


};

#endif // OLIVER_H