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
    inline int gather(BFS_Vertex src, EdgeId outdegree, Dump_Edge edge) {
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
    CommandLine cl(argc, argv, "<inFile> [-dimacs] [-verbose] [-round 100]");
    char * inFile = cl.getArgument(0);
    VertexId source = cl.getOptionIntValue("-s", 0);
    int max_rounds = cl.getOptionIntValue("-round", 100);
    bool dimacs = cl.getOption("-dimacs");
    bool verbose = cl.getOption("-verbose");
    int group_size = cl.getOptionIntValue("-g", 1);
    bool use_scan = cl.getOption("-scan");

    // Read the graph file.
    CsrGraph<int, int> graph;
    if (dimacs) {
        graph.fromDimacsFile(inFile);
    } else {
        graph.fromEdgeListFile(inFile);
    }

    // Algorithm specific parameters
    const int infCost = 0x7fffffff;

    Oliver<BFS_Vertex, Dump_Edge, int> ol;
    ol.readGraph(graph);

    // Initializes the value of all vertices.
    VertexSubset all(graph.vertexCount, true);
    ol.vertexMap<BFS_init_F>(all, BFS_init_F(infCost));
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
        int size = frontier.size();
        
        switch(group_size) {
            case 1:  ol.edgeFilter<BFS_edge_F, 1>(edgeFrontier, frontier, BFS_edge_F()); break;
            case 2:  ol.edgeFilter<BFS_edge_F, 2>(edgeFrontier, frontier, BFS_edge_F()); break;
            case 4:  ol.edgeFilter<BFS_edge_F, 4>(edgeFrontier, frontier, BFS_edge_F()); break;
            case 8:  ol.edgeFilter<BFS_edge_F, 8>(edgeFrontier, frontier, BFS_edge_F()); break;
            case 16: ol.edgeFilter<BFS_edge_F, 16>(edgeFrontier, frontier, BFS_edge_F()); break;
            case 32: ol.edgeFilter<BFS_edge_F, 32>(edgeFrontier, frontier, BFS_edge_F()); break;
            default: assert(0);
        }

        if (use_scan) 
            ol.vertexFilter<BFS_vertex_F, true>(frontier, edgeFrontier, BFS_vertex_F(infCost));
        else
            ol.vertexFilter<BFS_vertex_F, false>(frontier, edgeFrontier, BFS_vertex_F(infCost));
  
        if (size == 0 || iterations == max_rounds) break;
        if (verbose) {
            LOG(INFO) << "BFS iterations " << iterations <<", size: "<< size
                      <<", time: " << w.getElapsedMillis() << "ms";
        }
        iterations++;
    }

    double totalTime =  getTimeMillis() - start;
    LOG(INFO) << "iterations: "<< iterations <<", time: " << totalTime << "ms";

    // Log the vertex value into a file
    outputFile = fopen("BFS.txt", "w");
    ol.printVertices();

    frontier.del();
    edgeFrontier.del();
    return 0;
}
