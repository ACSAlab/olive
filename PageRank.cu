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
 * Distributed memory implementation using Olive framework.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2015-01-13
 * Last Modified: 2015-01-13
 */

#include "oliver.h"

FILE * outputFile;

struct PR_Vertex {
    double rank;
    double delta;

    void print() {
        fprintf(outputFile, "%f\t%f\n", rank, delta);
    }
};

struct PR_edge_F {
    __device__
    inline double gather(PR_Vertex srcValue, EdgeId outdegree) {
        return srcValue.rank / outdegree;
    }

    __device__
    inline void reduce(double &accumulator, double accum) {
        atomicAdd(&accumulator, accum);
    }
};  // EdgeMap

struct PR_vertex_F {
    double damping, fraction, addConstant;

    PR_vertex_F(double _damping, double _oneOverN, double _fraction) : 
        damping(_damping), fraction(_fraction),
        addConstant( (1-_damping) * _oneOverN ) {}

    __device__ bool cond(PR_Vertex v, VertexId id) {
        return (fabs(v.delta) > (fraction * v.rank));
    }

    __device__
    inline void update(PR_Vertex &v, double accum) {
        v.delta = damping * accum + addConstant - v.rank;
        v.rank += v.delta;
    }
};  // VertexFilter

struct PR_init_F {
    double rank;

    PR_init_F(double _rank): rank(_rank) {}

    __device__ bool cond(PR_Vertex v, VertexId id) {
        return true;
    }

    __device__
    inline void update(PR_Vertex &v, double accum) {
        v.rank = rank;
        v.delta = rank;
    }
};  // VertexMap

int main(int argc, char **argv) {
    
    CommandLine cl(argc, argv, "<inFile> [-max 100]");
    char * inFile = cl.getArgument(0);
    int maxIterations = cl.getOptionIntValue("-max", 100);

    // Read in the graph data.
    CsrGraph<int, int> graph;
    graph.fromEdgeListFile(inFile);
    Oliver<PR_Vertex, double> ol;
    ol.readGraph(graph);

    // Write the result.
    outputFile = fopen("PageRank.txt", "w");

    // Algorithm specific parameters
    const double damping = 0.85;
    const double oneOverN = 1.0 / ol.getVertexCount();
    const double fraction = 0.01;

    // Initialize all vertices rank value to 1/n, and activate them
    ol.vertexFilter<PR_init_F>(PR_init_F(oneOverN));

    double start = getTimeMillis();

    int i = 0;
    int qSize;
    while ((qSize = ol.getWorkqueueSize()) > 0) {
        if (i >= maxIterations) break;
        LOG(INFO) << "\nPR iterations " << i << " size: " << qSize;
        ol.edgeMap<PR_edge_F>(PR_edge_F());
        ol.vertexFilter<PR_vertex_F>(PR_vertex_F(damping, oneOverN, fraction));
        i++;
    }

    LOG(INFO) << "time=" << getTimeMillis() - start << "ms";

    ol.printVertices();

    return 0;
}


