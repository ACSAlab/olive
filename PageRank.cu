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
        fprintf(outputFile, "%f\n", rank);
    }

    void reduce(double &r) {
        r += fabs(delta);
    }
};

struct PR_edge_F {
    __device__
    inline double gather(PR_Vertex srcValue, EdgeId outdegree, int edgeValue) {
        return srcValue.rank / outdegree;
    }

    __device__
    inline void reduce(double &accumulator, double accum) {
        atomicAdd(&accumulator, accum);
    }
};  // edgeMa

struct PR_vertex_F {
    double damping, addConstant;

    PR_vertex_F(double _damping, double _oneOverN) : damping(_damping),
        addConstant( (1-_damping) * _oneOverN ) {}

    __device__
    inline void operator() (PR_Vertex &v, double accum) {
        double new_rank = damping * accum + addConstant;
        v.delta = new_rank - v.rank;
        v.rank = new_rank;
    }
};  // vertexMap

struct PR_init_F {
    double rank;

    PR_init_F(double _rank): rank(_rank) {}

    __device__
    inline void operator() (PR_Vertex &v, double accum) {
        v.rank = rank;
        v.delta = rank;
    }
};  // vertexMap

int main(int argc, char **argv) {
    
    CommandLine cl(argc, argv, "<inFile> [-dimacs] [-verbose]");
    char * inFile = cl.getArgument(0);
    bool dimacs = cl.getOption("-dimacs");
    bool verbose = cl.getOption("-verbose");

    // Read the graph file.
    CsrGraph<int, int> graph;
    if (dimacs) {
        graph.fromDimacsFile(inFile);
    } else {
        graph.fromEdgeListFile(inFile);
    }

    Oliver<PR_Vertex, double, int> ol;
    ol.readGraph(graph);

    // Algorithm specific parameters
    const double damping = 0.85;
    const double oneOverN = 1.0 / ol.getVertexCount();
    const double epsilon = 0.0000001;

    // Universal vertex set in sparse representation
    VertexSubset all(graph.vertexCount, true);  
    ol.vertexMap<PR_init_F>(all, PR_init_F(oneOverN));

    double start = getTimeMillis();
    Stopwatch w;
    w.start();

    int iterations = 0;
    while (1) {
        iterations++;

        ol.edgeMap<PR_edge_F>(all, PR_edge_F());
        ol.vertexMap<PR_vertex_F>(all, PR_vertex_F(damping, oneOverN));

        double err = ol.vertexReduce();

        if (verbose) LOG(INFO) << "PR iterations: " << iterations
                               << ", err: " << err
                               <<", time: " << w.getElapsedMillis() << "ms";
        if (err < epsilon) break;
    }

    LOG(INFO) << "iterations: "<< iterations 
              <<", time: " << getTimeMillis() - start << "ms";

    // Log the vertex value into a file
    outputFile = fopen("PageRank.txt", "w");
    ol.printVertices();

    all.del();

    return 0;
}


