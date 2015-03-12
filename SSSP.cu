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
 * Single-Source Shortest Path
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-12-18
 */

#include "oliver.h"

FILE * outputFile;

struct SSSP_Vertex {
    int distance;

    inline void print() {
        fprintf(outputFile, "%d\n", distance);
    }
};

struct SSSP_Edge {
    int weight;

    inline void print() {
        fprintf(outputFile, "%d\n", weight);
    }
};

struct SSSP_edge_F {
    __device__
    inline int gather(SSSP_Vertex src, EdgeId outdegree, SSSP_Edge edge) {
        return src.distance + edge.weight;        
    }

    __device__
    inline void reduce(int &accumulator, int accum) {
        atomicMin(&accumulator, accum);
    }
};  // edgeFilter

struct SSSP_vertex_F {
    __device__
    inline void update(SSSP_Vertex &v, int accum) {
        v.distance = accum;
    }

    __device__
    inline bool cond(SSSP_Vertex v, int accum) {
        return (v.distance > accum);
    }
};  // vertexFilter

struct SSSP_vertex_F_init {
    int distance;

    SSSP_vertex_F_init(int _d) : distance(_d) {}

    __device__
    inline void operator() (SSSP_Vertex &v, int accum) {
        v.distance = distance;
    }
};  // vertexMap


struct SSSP_edge_F_init {
    int weight;

    SSSP_edge_F_init(int _w) : weight(_w) {}

    __device__
    inline int gather(SSSP_Vertex src, EdgeId outdegree, SSSP_Edge &edge) {
        edge.weight = weight;
        return 0;       
    }

    __device__
    inline void reduce(int &accumulator, int accum) {}
};  // edgeMap to set the edge value


int main(int argc, char **argv) {
    CommandLine cl(argc, argv, "<inFile> [-s 0] [-dimacs] [-verbose]");
    char * inFile = cl.getArgument(0);
    VertexId source = cl.getOptionIntValue("-s", 0);
    int max_rounds = cl.getOptionIntValue("-round", 100);
    bool dimacs = cl.getOption("-dimacs");
    bool verbose = cl.getOption("-verbose");

    // Read the graph file.
    CsrGraph<int, int> graph;
    if (dimacs) {
        graph.fromDimacsFile(inFile);
    } else {
        graph.fromEdgeListFile(inFile);
    }

    // Algorithm specific parameters
    const int infDistance = 0x7fffffff;

    Oliver<SSSP_Vertex, SSSP_Edge, int> ol(infDistance);
    ol.readGraph(graph);

    // Initializes the value of all vertices and all edges with universal set.
    VertexSubset all(graph.vertexCount, true);
    ol.vertexMap<SSSP_vertex_F_init>(all, SSSP_vertex_F_init(infDistance));
    ol.edgeMap<SSSP_edge_F_init>(all, SSSP_edge_F_init(1));
    all.del();  // no longer used

    // Dense VertexSubset with a singleton vertex.
    VertexSubset frontier(graph.vertexCount, source); 
    ol.vertexMap<SSSP_vertex_F_init>(frontier, SSSP_vertex_F_init(0));

    // Sparse VertexSubset to represent the expanding edges.
    VertexSubset edgeFrontier(graph.vertexCount, false);

    double start = getTimeMillis();    
    Stopwatch w;
    w.start();

    int iterations = 0;
    while (1) {
        int size = frontier.size();
        ol.edgeFilter<SSSP_edge_F>(edgeFrontier, frontier, SSSP_edge_F());
        ol.vertexFilter<SSSP_vertex_F>(frontier, edgeFrontier, SSSP_vertex_F());

        if (size == 0 || iterations == max_rounds) break;

        // Detect the negative cycle
        if (iterations == graph.vertexCount) {
            LOG(INFO) << "negative cycle!";
            break;
        }
        if (verbose)
            LOG(INFO) << "SSSP iterations: " << iterations <<", size: " << size
                      << ", time: " << w.getElapsedMillis() << "ms";
        iterations++;

    }

    double totalTime =  getTimeMillis() - start;
    LOG(INFO) << "iterations: "<< iterations <<", time: " << totalTime << "ms";
    
    // Log the vertex value into a file
    outputFile = fopen("SSSP.txt", "w");
    ol.printVertices();

    frontier.del();
    edgeFrontier.del();
    return 0;
}