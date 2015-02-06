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


/**
 * The serial version is used to validate the correctness of the GPU version.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-12-17
 * Last Modified: 2014-12-18
 */

#include <vector>
#include <deque>

#define INF_COST 0x7fffffff

void expect_equal(std::vector<int> v1, std::vector<int> v2) {
    assert(v1.size() == v2.size());
    for (int i = 0; i < v1.size(); i++) {
        assert(v1[i] == v2[i]);
    }
}

/**
 * The following algorithm comes from CLRS.
 *
 * @param partition The graph partition
 * @param nodes     The number of nodes in the graph.
 * @param source    The source node to traverse from.
 * @return a vector containing the BFS level for each node.
 */
std::vector<int> bfs_serial(const CsrGraph<int, int> &graph, VertexId source) {

    GRD<int> levels;
    levels.reserve(graph.vertexCount);
    levels.allTo(INF_COST);

    GRD<int> visited;
    visited.reserve(graph.vertexCount);
    visited.allTo(0);

    std::deque<VertexId> current;
    current.push_back(source);
    levels.set(source, 0);
    visited.set(source, 1);

    while(!current.empty()) {
        VertexId v = current.front();
        current.pop_front();        // Dequeue
        for (EdgeId e = graph.srcVertices[v]; e < graph.srcVertices[v+1]; e ++) {
            VertexId dst = graph.outgoingEdges[e];
            if (visited[dst] == 0) {
                levels[dst] = levels[v] + 1; 
                current.push_back(dst);
                visited[dst] = 1;
            }
        }
    }

    return std::vector<int>(levels.elemsHost, levels.elemsHost + graph.vertexCount);
}