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

#include "olive.h"
#include "common.h"

#define INF_COST 0x7fffffff

__device__ int currentLevel_d;

__global__ void setCurrentLevel(int level) {
    currentLevel_d = level;
}

struct BFS_Vertex {
    int level;
};

struct BFS_edge_F {
    __device__
    inline bool gather(BFS_Vertex srcValue, EdgeId outdegree) {
        if (srcValue.level == currentLevel_d - 1) {
            printf("srcValue.level = %d\n", srcValue.level);
            printf("currentLevel_d = %d\n", currentLevel_d);
            return true;
        }
        return false;
    }

    __device__
    inline void reduce(bool &accumulator, bool accum) {
        if (accum) 
            accumulator = true;
    }
};

struct BFS_vertex_F {
    __device__
    inline bool cond(BFS_Vertex local) {
        return (local.level == INF_COST);
    }

    __device__
    inline void update(BFS_Vertex &local, bool accum) {
        if (accum) {
            printf ("change level\n\n");
            local.level = currentLevel_d;
        }
    }
};

struct BFS_init_F {
    int _level;
    BFS_init_F(int l): _level(l) {}

    __device__
    inline void update(BFS_Vertex &v, bool accum) {
        v.level = _level;
    }

    __device__ bool cond(BFS_Vertex v) {
        return true;
    }
};


static int *level_g;

struct BFS_at_F {
    inline void operator() (VertexId id, BFS_Vertex value) {
        level_g[id] = value.level;
    }
};

int main(int argc, char **argv) {

    CommandLine cl(argc, argv, "<inFile> [-rounds 10]");
    char * inFile = cl.getArgument(0);
    int rounds = cl.getOptionIntValue("-rounds", 10);

    int currentLevel;

    Olive<BFS_Vertex, bool> olive;
    olive.readGraph(inFile, 2);

    VertexId source = atoi(cl.getArgument(1));

    // The final result, which will be aggregated.
    level_g = new int[olive.getVertexCount()];

    // Initialize all vertices, and filter the source vertex
    olive.vertexFilterDense<BFS_init_F>(BFS_init_F(INF_COST));
    BFS_Vertex zero;
    zero.level = 0;
    olive.BFS_setSource<BFS_Vertex>(source, zero);    

    int iterations = 0;
    currentLevel = 1;
    while (!olive.allVerticesInactive() && iterations < 10) {
        cudaSetDevice(1);
        setCurrentLevel <<<1, 1>>> (currentLevel);
        cudaSetDevice(0);
        setCurrentLevel <<<1, 1>>> (currentLevel);
        // cudaMemcpyToSymbol("currentLevel_d", &currentLevel, sizeof(int), size_t(0), cudaMemcpyHostToDevice);
        // CUDA_CHECK(H2D(currentLevel_d, currentLevel, sizeof(int)));
        printf("\n\n\niterations: %d\n", iterations++);
        olive.edgeMapDense<BFS_edge_F>(BFS_edge_F());

        olive.vertexFilterDense<BFS_vertex_F>(BFS_vertex_F());
        currentLevel++;

        olive.vertexTransform<BFS_at_F>(BFS_at_F());

        for (int i = 0; i < olive.getVertexCount(); i++) {
            printf("%d ", level_g[i]);
        }
    }

    

    return 0;
}
