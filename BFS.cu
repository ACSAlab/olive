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
#include "unitestBFS.h"

#define INF_COST 0x7fffffff


struct BFS_Vertex {
    int level;
};

struct BFS_edge_F {
    __device__
    inline int gather(BFS_Vertex srcValue, EdgeId outdegree) {
        return srcValue.level + 1;        
    }

    __device__
    inline void reduce(int &accumulator, int accum) {
        accumulator = accum; // benign race happens
    }
};

struct BFS_vertex_F {
    __device__
    inline bool cond(BFS_Vertex local, VertexId id) {
        return (local.level == INF_COST);
    }

    __device__
    inline void update(BFS_Vertex &local, int accum) {
        local.level = accum;
    }
};


struct BFS_source_F {
    int _id;
    BFS_source_F(int id): _id(id) {}

    __device__ bool cond(BFS_Vertex v, VertexId id) {
        return (id == _id);
    }

    __device__
    inline void update(BFS_Vertex &v, int accum) {
        v.level = 0;
    }
};

struct BFS_init_F {
    __device__
    inline void update(BFS_Vertex &v) {
        v.level = INF_COST;
    }
};


int * levels_g;
struct BFS_gather_F {
    inline void operator() (int i, BFS_Vertex v) {
        levels_g[i] = v.level;
    }
};


int main(int argc, char **argv) {

    CommandLine cl(argc, argv, "<inFile> -s 2");
    char * inFile = cl.getArgument(0);
    int src = cl.getOptionIntValue("-s", 0);
    bool validate = cl.getOption("-validate");

    CsrGraph<int, int> graph;
    graph.fromEdgeListFile(inFile);


    Oliver<BFS_Vertex, int> olive;
    olive.readGraph(graph);

    olive.vertexMap<BFS_init_F>(BFS_init_F());
    olive.vertexFilter<BFS_source_F>(BFS_source_F(src));

    int iterations = 0;

    double start = getTimeMillis();

    while (olive.getWorkqueueSize() > 0) {
        // printf("\nBFS iterations %d\n", iterations);
        olive.edgeMap<BFS_edge_F>(BFS_edge_F());
        olive.vertexFilter<BFS_vertex_F>(BFS_vertex_F());
        //olive.print();
        iterations++;
    }

    LOG(INFO) << "time=" << getTimeMillis() - start << "ms";


    if (!validate) return;

    // Gather results
    levels_g = new int[olive.getVertexCount()];
    olive.vertexTransform<BFS_gather_F>(BFS_gather_F());
    auto result = std::vector<int>(levels_g, levels_g + olive.getVertexCount());

    // Call serial BFS
    auto serial_result = bfs_serial(graph, src);
    expect_equal(result, serial_result);
    printf("validated!\n");
    return 0;
}
