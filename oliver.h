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

template<typename VertexValue,
         typename AccumValue,
         typename EdgeValue>
class Oliver {
public:
    /**
     * The edgeFilter function invokes the corresponding subroutine
     * according to the representation of the source vertex subset and
     * produces a sparse vertex subset as output.
     */
    template<typename F>
    void edgeFilter(VertexSubset &dst, const VertexSubset &src, F f) {

        assert(!dst.isDense);

        // Clear the accumulator before the gather phase starts
        accumulators.clear();

        if (src.isDense) {
            auto c = util::kernelConfig(src.size());
            edgeFilterDenseKernel<VertexValue, AccumValue, EdgeValue, F>
            <<< c.first, c.second>>>(
                src.workqueue.elemsDevice,
                src.qSizeDevice,
                srcVertices.elemsDevice,
                outgoingEdges.elemsDevice,
                vertexValues.elemsDevice,
                accumulators.elemsDevice,
                edgeValues.elemsDevice,
                dst.workset.elemsDevice,
                f);
        } else {
            auto c = util::kernelConfig(src.capacity());
            edgeFilterSparseKernel<VertexValue, AccumValue, EdgeValue, F>
            <<< c.first, c.second>>>(
                src.workset.elemsDevice,
                src.capacity(),
                srcVertices.elemsDevice,
                outgoingEdges.elemsDevice,
                vertexValues.elemsDevice,
                accumulators.elemsDevice,
                edgeValues.elemsDevice,
                dst.workset.elemsDevice,
                f);
        }
        CUDA_CHECK(cudaThreadSynchronize());
    }

    /**
     * edgeMap is used to update the local edge state.
     * @param src  A subset of vertices the UDF will be applied to.
     * @param f    The UDF applied to the edge.
     */
    template<typename F>
    void edgeMap(const VertexSubset &src, F f) {

      // Clear the accumulator before the gather phase starts
        accumulators.clear();

        assert(!src.isDense);
        auto c = util::kernelConfig(src.capacity());
        edgeMapKernel<VertexValue, AccumValue, EdgeValue, F>
        <<< c.first, c.second>>>(
            src.workset.elemsDevice,
            src.capacity(),
            srcVertices.elemsDevice,
            outgoingEdges.elemsDevice,
            vertexValues.elemsDevice,
            accumulators.elemsDevice,
            edgeValues.elemsDevice,
            f);
        CUDA_CHECK(cudaThreadSynchronize());
    }


    /**
     * vertexFilter is used to update the local vertex state.
     *
     * It takes as input a sparse vertex subset and produces either a sparse
     * vertex subset or a dense one as output.
     */
    template<typename F>
    void vertexFilter(VertexSubset &dst, const VertexSubset &src, F f) {
        assert(!src.isDense);

        if (dst.isDense) {
            auto c = util::kernelConfig(src.capacity());
            vertexFilterDenseKernel<VertexValue, AccumValue, F>
            <<< c.first, c.second>>>(
                src.workset.elemsDevice,
                src.capacity(),
                vertexValues.elemsDevice,
                accumulators.elemsDevice,
                dst.workqueue.elemsDevice,
                dst.qSizeDevice,
                f);
        } else {
            auto c = util::kernelConfig(src.capacity());
            vertexFilterSparseKernel<VertexValue, AccumValue, F>
            <<< c.first, c.second>>>(
                src.workset.elemsDevice,
                src.capacity(),
                vertexValues.elemsDevice,
                accumulators.elemsDevice,
                dst.workset.elemsDevice,
                f);
        }
        CUDA_CHECK(cudaThreadSynchronize());
    }

    /**
     * vertexMap is used to update the local vertex state.
     * @param src  A subset of vertices the UDF will be applied to.
     * @param f    The UDF applied to the vertices.
     */
    template<typename F>
    void vertexMap(const VertexSubset &src, F f) {
        if (src.isDense) {
            auto c = util::kernelConfig(src.size());
            vertexMapDenseKernel<VertexValue, AccumValue, F>
            <<< c.first, c.second>>>(
                src.workqueue.elemsDevice,
                src.qSizeDevice,
                vertexValues.elemsDevice,
                accumulators.elemsDevice,
                f);
        } else {
            auto c = util::kernelConfig(src.capacity());
            vertexMapSparseKernel<VertexValue, AccumValue, F>
            <<< c.first, c.second>>>(
                src.workset.elemsDevice,
                src.capacity(),
                vertexValues.elemsDevice,
                accumulators.elemsDevice,
                f);
        }
        CUDA_CHECK(cudaThreadSynchronize());
    }



    /**
     * Reduce the vertex value by specifying a reduce function
     * @return The reduced result
     */
    AccumValue vertexReduce() {
        vertexValues.persist();
        AccumValue r = (AccumValue) 0;
        for (int i = 0; i < vertexCount; i++) {
            vertexValues[i].reduce(r);
        }
        return r;
    }

    void readGraph(const CsrGraph<int, int> &graph) {
        vertexCount = graph.vertexCount;
        edgeCount = graph.edgeCount;
        srcVertices.reserve(vertexCount + 1);
        outgoingEdges.reserve(edgeCount);
        vertexValues.reserve(vertexCount);
        accumulators.reserve(vertexCount);
        edgeValues.reserve(edgeCount);
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
        edgeValues.del();
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
    GRD<EdgeValue>   edgeValues;


};

#endif // OLIVER_H