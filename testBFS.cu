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

#include <deque>

#include "csrGraph.h"
#include "commandLine.h"
#include "timer.h"
/**
 * The following algorithm comes from CLRS.
 */
int main(int argc, char **argv) {

    CommandLine cl(argc, argv, "<inFile> -s 0");
    char * inFile = cl.getArgument(0);
    int source = cl.getOptionIntValue("-s", 0);
    CsrGraph<int, int> graph;
    graph.fromEdgeListFile(inFile);

    const int infiniteCost = 0x7fffffff;

    int * levels;
    levels = new int[graph.vertexCount];
    for (int i = 0; i < graph.vertexCount; i++) {
        levels[i] = infiniteCost;
    }
    levels[source]= 0;

    std::deque<VertexId> frontier;
    frontier.push_back(source);

    double start = getTimeMillis();

    while(!frontier.empty()) {
        VertexId v = frontier.front();
        frontier.pop_front();        // Dequeue
        for (EdgeId e = graph.vertices[v]; e < graph.vertices[v+1]; e++) {
            VertexId dst = graph.edges[e];
            if (levels[dst]  == infiniteCost) {
                levels[dst] = levels[v] + 1; 
                frontier.push_back(dst);
            }
        }
    }

    LOG(INFO) << "time=" << getTimeMillis() - start << "ms";

    FILE * outputFile;
    outputFile = fopen("BFS.serial.txt", "w");
    for (int i = 0; i < graph.vertexCount; i++) {
        fprintf(outputFile, "%d\n", levels[i]);
    }
}