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
 * CSR graph representation.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-13
 * Last Modified: 2015-02-05
 */

#ifndef CSR_GRAPH_H
#define CSR_GRAPH_H

#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <iostream>

#include "common.h"
#include "edgeTuple.h"
#include "partitionStrategy.h"
#include "logging.h"
#include "utils.h"
#include "timer.h"


/**
 * Flexible graph representation for in-memory querying or manipulating.
 */
template <typename VertexValue, typename EdgeValue>
class CsrGraph {
public:
    /** Total number of the vertices and edges in the graph. */
    VertexId     vertexCount;
    EdgeId       edgeCount;
 
    /** Buffers for CSR-formatted graph (as outgoing edges). */
    EdgeId      *srcVertices;
    VertexId    *outgoingEdges;
    EdgeValue   *outgoingEdgeValues;

    /** Buffers for CSR-formatted graph (as incoming edges). */
    EdgeId      *dstVertices;
    VertexId    *incomingEdges;
    EdgeValue   *incomingEdgeValues;

    /** Constructor */
    CsrGraph(): vertexCount(0), edgeCount(0),
        srcVertices(NULL), dstVertices(NULL),
        outgoingEdges(NULL), incomingEdges(NULL),
        outgoingEdgeValues(NULL), incomingEdgeValues(NULL) {}

    ~CsrGraph() {
        if (srcVertices) delete srcVertices;
        if (dstVertices) delete dstVertices;
        if (outgoingEdges) delete outgoingEdges;
        if (incomingEdges) delete incomingEdges;
        if (outgoingEdgeValues) delete outgoingEdgeValues;
        if (incomingEdgeValues) delete incomingEdgeValues;
    }

    void initGraph(VertexId vertices, EdgeId edges) {
        vertexCount = vertices;
        edgeCount = edges;
        // TODO: here just ignore the values
        srcVertices = new EdgeId[vertexCount + 1];
        dstVertices = new EdgeId[vertexCount + 1];
        outgoingEdges = new VertexId[edgeCount];
        incomingEdges = new VertexId[edgeCount];
    }

    /**
     * Load the graph from an edge list file where each line contains two
     * integers: a source id and a target id. Skips lines that begin with `#`.
     *
     * @note If a graph is loaded from  a edge list file, the type vertex value
     * is `int`. Meanwhile, the vertex-associated data is given a meaningless
     * value (0 by default).
     *
     * @example Loads a file in the following format:
     * {{{
     * # Comment Line
     * # Source Id  Target Id
     * 1    5
     * 1    2
     * 2    7
     * 1    8
     * }}}
     *
     * @param path The path to the graph
     */
    void fromEdgeListFile(const char *path) {
        FILE *file = fopen(path, "r");
        if (file == NULL) {
            LOG(ERROR) << "Can not open graph file: " << path;
            return;
        }

        Stopwatch stopwatch;
        stopwatch.start();

        char line[1024];
        typedef EdgeTuple<int> EdgeTupleInt;
        EdgeTupleInt *parsedEdgeTuples; // Buffer for the parsed edge tuples.
        EdgeId parsedEdges = VertexId(-1);

        while (fscanf(file, "%[^\n]\n", line) > 0) {
            switch (line[0]) {
            case '#': // Skip the comment
                break;
            default:
                if (parsedEdges == VertexId(-1)) {
                    long long llnodes, lledges;
                    sscanf(line, "%lld %lld", &llnodes, &lledges);
                    LOG(INFO) << "Parsing " << llnodes << " nodes, " << lledges << " edges";
                    assert(llnodes > 0);
                    parsedEdgeTuples = (EdgeTupleInt *) malloc(lledges * sizeof(EdgeTupleInt));
                    initGraph(llnodes, lledges);
                    parsedEdges = 0;
                } else {
                    long long llsrc, lldst;
                    sscanf(line, "%lld %lld", &llsrc, &lldst);
                    parsedEdgeTuples[parsedEdges] = EdgeTupleInt(llsrc, lldst, 1);
                    parsedEdges++;
                }
            }
        }

        LOG(INFO) << "It took " << stopwatch.getElapsedMillis()
                  << "ms to parse " << parsedEdges << " edge tuples.";

        // Generate the outgoing edge list first.
        // Clustering the edge tuples by src Id.
        std::stable_sort(parsedEdgeTuples, parsedEdgeTuples + edgeCount,
                         edgeTupleSrcCompare<EdgeTupleInt>);

        VertexId prevSrc = VertexId(-1);
        for (EdgeId edge = 0; edge < edgeCount; edge++) {
            outgoingEdges[edge] = parsedEdgeTuples[edge].dstId;
            VertexId src = parsedEdgeTuples[edge].srcId;
            // Fill up the missing srcs
            for (VertexId v = prevSrc + 1; v <= src; v++) {
                srcVertices[v] = edge;
            }
            prevSrc = src;
        }

        // Fill out trailing vertices whose has not connecting edges.
        for (VertexId v = prevSrc + 1; v <= vertexCount; v++) {
            srcVertices[v] = edgeCount;
        }

        // Generate the incoming edge list.
        // Clustering the edge tuples by dst Id.
        std::stable_sort(parsedEdgeTuples, parsedEdgeTuples + edgeCount,
                         edgeTupleDstCompare<EdgeTupleInt>);

        VertexId prevDst = VertexId(-1);
        for (EdgeId edge = 0; edge < edgeCount; edge++) {
            VertexId dst = parsedEdgeTuples[edge].dstId;
            incomingEdges[edge] = parsedEdgeTuples[edge].srcId;
            // Fill up the missing dsts
            for (VertexId v = prevDst + 1; v <= dst; v++) {
                dstVertices[v] = edge;
            }
            prevDst = dst;
        }

        // Fill out trailing vertices whose has not connecting edges.
        for (VertexId v = prevDst + 1; v <= vertexCount; v++) {
            dstVertices[v] = edgeCount;
        }

        LOG(INFO) << "It took " << stopwatch.getElapsedMillis()
                  << "ms to generate the CSR graph.";

        if (parsedEdgeTuples) free(parsedEdgeTuples);
    }

    /**
     * Print the graph on the screen as the outgoing edges.
     */
    void printOutEdges(bool withValue = false) const {
        for (VertexId src = 0; src < vertexCount; src++) {
            std::cout << "[" << src << "] ";
            for (EdgeId e = srcVertices[src]; e < srcVertices[src + 1]; e++) {
                std::cout << " ->" << outgoingEdges[e];
            }
            std::cout << std::endl;
        }
    }

    /**
     * Print the graph on the screen as the outgoing edges.
     */
    void printInEdges(bool withValue = false) const {
        for (VertexId dst = 0; dst < vertexCount; dst++) {
            std::cout << "[" << dst << "] ";
            for (EdgeId e = dstVertices[dst]; e < dstVertices[dst + 1]; e++) {
                std::cout << " <-" << incomingEdges[e];
            }
            std::cout << std::endl;
        }
    }

    /**
     * Prints the degree distribution in log-style on the screen.
     * e.g., the output will look:
     * {{{
     * degreeLog[0]: 0         5%
     * degreeLog[1]: (1, 2]   15%
     * degreeLog[2]: (2, 4]   29%
     * ...
     * }}}
     */
    void printDegreeHistogram(bool outdegree = true) {
        size_t degLog[32];
        int slotMax = 0;
        for (int i = 0; i < 32; i++) {
            degLog[i] = 0;
        }

        for (VertexId v = 0; v < vertexCount; v++) {
            EdgeId deg;
            if (outdegree) {
                deg = srcVertices[v + 1] - srcVertices[v];
            } else {
                deg = dstVertices[v + 1] - dstVertices[v];
            }

            int slot = 0;
            while (deg > 0) {
                deg /= 2;
                slot++;
            }
            if (slot > slotMax) {
                slotMax = slot;
            }
            degLog[slot]++;
        }

        printf("deg[0]: 0\t%llu\t%.2f%%\n", degLog[0], (float) degLog[0] / vertexCount * 100);
        for (int i = 1; i <= slotMax; i++) {
            int high = pow(2, i - 1);
            int low  = pow(2, i - 2);
            printf("deg[%d]: (%d, %d]\t%llu\t%.2f%%\n", i, low, high,
                   degLog[i], (float) degLog[i] / vertexCount * 100);
        }
    }
};


#endif  // CSR_GRAPH_H
