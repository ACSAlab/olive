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
    int maxIterations = cl.getOptionIntValue("-max", 100);
    CsrGraph<int, int> graph;
    graph.fromEdgeListFile(inFile);


    const double damping = 0.85;
    const double oneOverN = 1.0 /  graph.vertexCount;

    GRD<double> ranks;
    ranks.reserve(graph.vertexCount);
    ranks.allTo(oneOverN);

    GRD<double> nghSums;
    nghSums.reserve(graph.vertexCount);
    nghSums.allTo(0);

    int i = 0;
    while (i < maxIterations) {
        for (VertexId v = 0; v < graph.vertexCount; v++) {
            for (EdgeId e = graph.srcVertices[v]; e < graph.srcVertices[v + 1]; e ++) {
                EdgeId outdeg = graph.srcVertices[v+1] - graph.srcVertices[v];
                VertexId dst = graph.outgoingEdges[e];
                nghSums[dst] += ranks[v] / outdeg;
            }
        }

        for (VertexId v = 0; v < graph.vertexCount; v++) {
            ranks[v] = damping * nghSums[v] + (1 - damping) * oneOverN;
        }

        nghSums.allTo(0);
        i++;
    }

    FILE * outputFile;
    outputFile = fopen("PageRank.serial.txt", "w");
    for (int i = 0; i < graph.vertexCount; i++) {
        fprintf(outputFile, "%f\n", ranks[i]);
    }

    return 0;
}