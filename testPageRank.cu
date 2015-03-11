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
 * The serial version is used to validate the correctness of the GPU version.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-12-17
 * Last Modified: 2014-12-18
 */


#include "csrGraph.h"
#include "commandLine.h"
#include "grd.h"

/**
 * The following Page Rank algorithm is controlled by specifying iterations.
 */
int main(int argc, char **argv) {

    CommandLine cl(argc, argv, "<inFile> [-max 100]");
    char * inFile = cl.getArgument(0);
    int maxIterations = cl.getOptionIntValue("-max", 1000);
    bool dimacs = cl.getOption("-dimacs");
    bool verbose = cl.getOption("-verbose");

    CsrGraph<int, int> graph;
    if (dimacs) {
        graph.fromDimacsFile(inFile);
    } else {
        graph.fromEdgeListFile(inFile);
    }


    const double damping = 0.85;
    const double oneOverN = 1.0 /  graph.vertexCount;
    const double epsilon = 0.0000001;

    double * ranks = new double[graph.vertexCount];
    double * deltas = new double[graph.vertexCount];
    double * nghSums = new double[graph.vertexCount];
    for (int i = 0; i < graph.vertexCount; i++) {
        ranks[i] = oneOverN;
        deltas[i] = oneOverN;
        nghSums[i] = 0;
    }

    double start = getTimeMillis();
    Stopwatch w;
    w.start();

    int iterations = 0;
    while (iterations < maxIterations) {
        for (int i = 0; i < graph.vertexCount; i++) {
            nghSums[i] = 0;
        }
        for (VertexId v = 0; v < graph.vertexCount; v++) {
            for (EdgeId e = graph.vertices[v]; e < graph.vertices[v + 1];e ++) {
                EdgeId outdeg = graph.vertices[v+1] - graph.vertices[v];
                VertexId dst = graph.edges[e];
                nghSums[dst] += (ranks[v] / outdeg);
            }
        }

        for (VertexId v = 0; v < graph.vertexCount; v++) {
            double new_rank = damping * nghSums[v] + (1 - damping) * oneOverN;
            deltas[v] = new_rank - ranks[v];
            ranks[v] = new_rank;
        }

        double err = 0.0;
        for (int i = 0; i < graph.vertexCount; i++) {
            err += fabs(deltas[i]);
        }

        if (verbose) LOG(INFO) << "PR iterations: " << iterations
                               << ", err: " << err
                               <<", time: " << w.getElapsedMillis() << "ms";

        if (err < epsilon) break;
        iterations++;
    }

    LOG(INFO) << "iterations="<< iterations 
              <<", time=" << getTimeMillis() - start << "ms";

    FILE * outputFile;
    outputFile = fopen("PageRank.serial.txt", "w");
    for (int i = 0; i < graph.vertexCount; i++) {
        fprintf(outputFile, "%f\n", ranks[i]);
    }

    return 0;
}