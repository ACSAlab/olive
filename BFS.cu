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
    inline int gather(BFS_Vertex src, EdgeId outdegree, int edgeValue) {
        return src.level + 1;        
    }

    __device__
    inline void reduce(int &accumulator, int accum) {
        accumulator = accum; // benign race happens
    }
};  // edgeMap

struct BFS_vertex_F {
    int infiniteCost;

    BFS_vertex_F(int _inf) : infiniteCost(_inf) {}

    __device__
    inline void update(BFS_Vertex &v, int accum) { v.level = accum; }

    __device__
    inline bool cond(BFS_Vertex v, int accum) {
        return (v.level == infiniteCost);
    }
};  // vertexFilter

struct BFS_init_F {
    int level;

    BFS_init_F(int _level) : level(_level) {}

    __device__
    inline void operator() (BFS_Vertex &v, int accum) { v.level = level; }
};  // vertexMap


int main(int argc, char **argv) {

    CommandLine cl(argc, argv, "<inFile> [-s 0] [-dimacs] [-verbose]");
    char * inFile = cl.getArgument(0);
    VertexId source = cl.getOptionIntValue("-s", 0);
    bool dimacs = cl.getOption("-dimacs");
    bool verbose = cl.getOption("-verbose");

    // Read the graph file.
    CsrGraph<int, int> graph;
    if (dimacs) {
        graph.fromDimacsFile(inFile);
    } else {
        graph.fromEdgeListFile(inFile);
    }

    Oliver<BFS_Vertex, int, int> ol;
    ol.readGraph(graph);

    // Algorithm specific parameters
    const int infiniteCost = 0x7fffffff;

    // Initializes the value of all vertices.
    VertexSubset all(graph.vertexCount, true);
    ol.vertexMap<BFS_init_F>(all, BFS_init_F(infiniteCost));
    all.del();  // No longer used

    // Make a dense VertexSubset with a singleton vertex (source)
    // and initializes the level of it to 0
    VertexSubset frontier(graph.vertexCount, source);
    ol.vertexMap<BFS_init_F>(frontier, BFS_init_F(0));

    // Sparse VertexSubset to represent the expanding edges.
    VertexSubset edgeFrontier(graph.vertexCount, false); 

    double start = getTimeMillis();    
    Stopwatch w;
    w.start();

    int iterations = 0;
    while (1) {
        iterations++;

        ol.edgeFilter<BFS_edge_F>(edgeFrontier, frontier, BFS_edge_F());
        frontier.clear();
        ol.vertexFilter<BFS_vertex_F>(frontier, edgeFrontier, BFS_vertex_F(infiniteCost));
        edgeFrontier.clear();
        if (frontier.size() == 0) break;
        if (verbose) LOG(INFO) << "BFS iterations " << iterations
                               <<", size: " << frontier.size()
                               <<", time: " << w.getElapsedMillis() << "ms";
    }

    LOG(INFO) << "time=" << getTimeMillis() - start << "ms";

    // Log the vertex value into a file
    outputFile = fopen("BFS.txt", "w");
    ol.printVertices();

    frontier.del();
    edgeFrontier.del();
    return 0;
}
