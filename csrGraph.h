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
 * CSR graph representation.
 */
template <typename VertexValue, typename EdgeValue>
class CsrGraph {
public:
    /** Total number of the vertices and edges in the graph. */
    VertexId     vertexCount;
    EdgeId       edgeCount;

    /** Buffers for CSR-formatted graph. */
    EdgeId      *vertices;
    VertexId    *edges;
    EdgeValue   *edgeValues;
    VertexValue *vertexValues;

    CsrGraph(): vertexCount(0), edgeCount(0), vertices(NULL), edges(NULL),
        edgeValues(NULL), vertexValues(NULL) {}

    ~CsrGraph() {
        if (vertices) delete vertices;
        if (edges) delete edges;
        if (edgeValues) delete edgeValues;
        if (vertexValues) delete vertexValues;
    }

    void initGraph(VertexId _vertexCount, EdgeId _edgeCount) {
        vertexCount = _vertexCount;
        edgeCount = _edgeCount;
        vertices = new EdgeId[vertexCount + 1];
        edges = new VertexId[edgeCount];
        edgeValues = new EdgeValue[edgeCount];
        vertexValues = new VertexValue[vertexCount];
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
        EdgeId parsedEdges = VertexId(-1);
        long long llnodes, lledges;

        // Cache the parsed edge tuples in the buffer.
        EdgeTupleInt *tuples;

        while (fscanf(file, "%[^\n]\n", line) > 0) {
            switch (line[0]) {
            case '#': // Skip the comment
                break;
            default:
                if (parsedEdges == VertexId(-1)) {
                    sscanf(line, "%lld %lld", &llnodes, &lledges);
                    LOG(INFO) << "Parsing graph: " << llnodes << " nodes, " << lledges << " edges";
                    assert(llnodes > 0);
                    tuples = (EdgeTupleInt *) malloc(lledges * sizeof(EdgeTupleInt));
                    initGraph(llnodes, lledges);
                    parsedEdges = 0;
                } else {
                    long long llsrc, lldst;
                    sscanf(line, "%lld %lld", &llsrc, &lldst);
                    tuples[parsedEdges] = EdgeTupleInt(llsrc, lldst, 1);
                    parsedEdges++;
                }
            }
        }

        LOG(INFO) << "It took " << stopwatch.getElapsedMillis()
                  << "ms to parse " << parsedEdges << " edge tuples.";

        // Generate the edge list by clustering the edge tuples by src Id.
        // std::stable_sort(tuples, tuples + edgeCount, edgeTupleSrcCompare<EdgeTupleInt>);

        VertexId prevSrc = VertexId(-1);
        for (EdgeId e = 0; e < edgeCount; e++) {
            edges[e] = tuples[e].dstId;
            VertexId src = tuples[e].srcId;
            // Fill up the missing `src`s
            for (VertexId v = prevSrc + 1; v <= src; v++) {
                vertices[v] = e;
            }
            prevSrc = src;
        }

        // Fill out trailing vertices whose has not connecting edges.
        for (VertexId v = prevSrc + 1; v <= vertexCount; v++) {
            vertices[v] = edgeCount;
        }

        if (tuples) free(tuples);

        LOG(INFO) << "It took " << stopwatch.getElapsedMillis()
                  << "ms to generate the CSR graph from edge list file.";
    }


    /**
     * From dimacs graph format
     */
    void fromDimacsFile(const char *path) {
        FILE *file = fopen(path, "r");
        if (file == NULL) {
            LOG(ERROR) << "Can not open graph file: " << path;
            return;
        }

        Stopwatch stopwatch;
        stopwatch.start();

        char line[1024];
        char  c;
        EdgeId parsededges = 0;
        VertexId parsedVertices = VertexId(-1);
        long long llnodes, lledges, lldstId;

        while ((c = fgetc(file)) != EOF) {
            switch (c) {
            case '#': // comment: skip any char encountered until see a '\n'
                while ((c = fgetc(file)) != EOF) {
                    if (c == '\n') break;
                }
                break;
            case ' ':
            case '\t': // white space
                break;
            case '\n':
                // end of line: begin to process the next node
                parsedVertices++;
                vertices[parsedVertices] = parsededges;
                break;
            default:
                ungetc(c, file); // put the char back
                if (parsedVertices == VertexId(-1)) {
                    fscanf(file, "%lld %lld[^\n]", &llnodes, &lledges, line);
                    initGraph(llnodes, lledges * 2);
                    LOG(INFO) << "Parsing graph: " << llnodes << " nodes, " << lledges << " (bi)edges";
                } else {
                    fscanf(file, "%lld", &lldstId); // process next edge in the same row
                    // The ids for dimacs of start at 1
                    edges[parsededges] = lldstId - 1;
                    parsededges++;
                }
            } // end of switch
        }

        // Fill out any trailing rows that didn't have explicit lines
        while (parsedVertices < vertexCount) {
            parsedVertices++;
            vertices[parsedVertices] = parsededges;
        }

        if (parsededges != edgeCount) {
            LOG(ERROR) << parsededges << "!=" << edgeCount;
            assert(0);
        }

        LOG(INFO) << "It took " << stopwatch.getElapsedMillis()
                  << "ms to generate the CSR graph from Dimacs file.";
    }

    /**
     * Print the graph on the screen as the outgoing edges.
     */
    void print(bool verbose = false) const {
        if (verbose) {
            for (VertexId v = 0; v < vertexCount; v++) {
                printf("[%lld] ", v);
                for (EdgeId e = vertices[v]; e < vertices[v + 1]; e++) {
                    std::cout << " ->" << edges[e];
                    printf(" ->%lld", edges[e]);
                }
                printf("\n");
            }
        }

        // Prints the degree distribution in log-style on the screen.
        // e.g., the output will look like:
        // degreeLog[0]: 0         5%
        // degreeLog[1]: (1, 2]   15%
        // degreeLog[2]: (2, 4]   29%
        size_t degLog[32];
        int slotMax = 0;
        for (int i = 0; i < 32; i++) {
            degLog[i] = 0;
        }

        EdgeId deg;
        for (VertexId v = 0; v < vertexCount; v++) {
            deg = vertices[v + 1] - vertices[v];
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
