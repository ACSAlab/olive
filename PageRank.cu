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

#include "olive.h"

#define DAMPING 0.85
#define EPSILON 0.000002

struct PR_Vertex {
    float rank;
    float delta;
};

struct PR_edge_F {
    __device__
    inline float gather(PR_Vertex srcValue, EdgeId outdegree) {
        return srcValue.rank / outdegree;
    }

    __device__
    inline void reduce(float &accumulator, float accum) {
        atomicAdd(&accumulator, accum);
    }
};

struct PR_vertex_F {
    __device__ bool cond(PR_Vertex v) {
        return (fabs(v.delta) > EPSILON);
    }

    __device__
    inline void update(PR_Vertex &v, float accum) {
        float rank_new = (1-DAMPING) + DAMPING * accum;
        v.delta = rank_new - v.rank;
        v.rank = rank_new;
    }
};


struct PR_init_F {
    float _rank;
    PR_init_F(float r): _rank(r) {}

    __device__
    inline void update(PR_Vertex &v) {
        v.rank = _rank;
        v.delta = _rank;
    }

    __device__ bool cond(PR_Vertex v) {
        return true;
    }
};

static float *ranks_g;
static float *deltas_g;



struct PR_at_F {
    inline void operator() (VertexId id, PR_Vertex v) {
        ranks_g[id] = v.rank;
        deltas_g[id] = v.delta;
    }
};


int main(int argc, char **argv) {

    CommandLine cl(argc, argv, "<inFile> [-rounds 20]");
    char * inFile = cl.getArgument(0);
    int rounds = cl.getOptionIntValue("-rounds", 20);

    Olive<PR_Vertex, float> olive;
    olive.readGraph(inFile, 2);

    // The final result, which will be aggregated.
    ranks_g = new float[olive.getVertexCount()];
    deltas_g = new float[olive.getVertexCount()];


    // Initialize all vertices rank value to 1/n, and activate them
    olive.vertexFilter<PR_init_F>(PR_init_F(1.0 /  olive.getVertexCount()));

    int i = 0;
    while (olive.getWorkqueueSize() > 0) {
        if (i >= rounds) break;
        printf("\n\n\niterations %d\n", i++);
        olive.edgeMap<PR_edge_F>(PR_edge_F());

        olive.vertexTransform<PR_at_F>(PR_at_F());
        for (int i = 0; i < olive.getVertexCount(); i++) {
            printf("%d %f %f\n", i, ranks_g[i], deltas_g[i]);
        }

        olive.vertexMap<PR_vertex_F>(PR_vertex_F());
        olive.vertexTransform<PR_at_F>(PR_at_F());
        for (int i = 0; i < olive.getVertexCount(); i++) {
            printf("%d %f %f\n", i, ranks_g[i], deltas_g[i]);
        }


    }

    // olive.vertexTransform<PR_at_F>(PR_at_F());
    // for (int i = 0; i < olive.getVertexCount(); i++) {
    //     printf("%d %f %f\n", i, ranks_g[i], deltas_g[i]);
    // }

    return 0;
}


