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
 * Test the bfs implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-12-18
 */

#include "oliver.h"

FILE * outputFile;

struct BFS_Vertex {
    int level;

    inline void print() {
        fprintf(outputFile, "%d\n", level);
    }
};

struct BFS_edge_F {
    __device__
    inline int gather(BFS_Vertex src, EdgeId outdegree) {
        return src.level + 1;        
    }

    __device__
    inline void reduce(int &accumulator, int accum) {
        accumulator = accum; // benign race happens
    }
};  // EdgeMap

struct BFS_vertex_F {
    int infiniteCost;

    BFS_vertex_F(int _inf) : infiniteCost(_inf) {}

    __device__
    inline bool cond(BFS_Vertex v, VertexId id) {
        return (v.level == infiniteCost);
    }

    __device__
    inline void update(BFS_Vertex &v, int accum) {
        v.level = accum;
    }
};  // VertexFilter

struct BFS_init_F {
    int infiniteCost;
    VertexId srcId;

    BFS_init_F(int _cost, VertexId _id) : infiniteCost(_cost), srcId(_id) {}

    __device__
    inline void update(BFS_Vertex &v, VertexId id) {
        if (id == srcId) {
            v.level = 0;
        } else {
            v.level = infiniteCost;
        }
    }
};  // VertexMap

int main(int argc, char **argv) {

    CommandLine cl(argc, argv, "<inFile> -s 0");
    char * inFile = cl.getArgument(0);
    int source = cl.getOptionIntValue("-s", 0);
    bool verbose = cl.getOption("-verbose");

    // Read in the graph data.
    CsrGraph<int, int> graph;
    graph.fromEdgeListFile(inFile);
    Oliver<BFS_Vertex, int> ol;
    ol.readGraph(graph);

    // Write the result.
    outputFile = fopen("BFS.txt", "w");

    // Algorithm specific parameters
    const int infiniteCost = 0x7fffffff;

    ol.vertexMap<BFS_init_F>(BFS_init_F(infiniteCost, source));

    // Dense representation
    VertexSubset frontier(graph.vertexCount, source, true);

    // Sparse representation
    VertexSubset edgeFontier(graph.vertexCount, false);

    double start = getTimeMillis();    
    Stopwatch w;
    w.start();

    int frontierSize;
    int i = 0;
    while ((frontierSize = frontier.size()) >0) {
        
        // frontier.print();
        ol.edgeMap<BFS_edge_F>(edgeFontier, frontier, BFS_edge_F());
        frontier.clear();

        // edgeFontier.print();
        ol.vertexFilter<BFS_vertex_F>(frontier, edgeFontier, BFS_vertex_F(infiniteCost));
        edgeFontier.clear();

        if (verbose)
            LOG(INFO) << "BFS iterations " << i++ << ", size " << frontierSize
                      <<", time=" << w.getElapsedMillis() << "ms";
    }

    LOG(INFO) << "time=" << getTimeMillis() - start << "ms";

    ol.printVertices();

    frontier.del();

    return 0;
}
