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
#define EPSILON 0.0000001

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
    __device__
    inline void operator() (PR_Vertex &v, float accum) {
        // printf("accum: %f\n", accum);
        float rank_new = (1-DAMPING) + DAMPING * accum;
        v.delta = rank_new - v.rank;
        v.rank = rank_new;
    }
};


struct PR_init_F {
    float _rank;
    PR_init_F(float r): _rank(r) {}

    __device__
    inline void operator() (PR_Vertex &v, float accum) {
        v.rank = _rank;
    }
};

static float *ranks_g;

struct PR_at_F {
    inline void operator() (VertexId id, PR_Vertex v) {
        ranks_g[id] = v.rank;
    }
};


int main(int argc, char **argv) {

    if (argc < 2) {
        printf("wrong argument");
        return 1;
    }

    Olive<PR_Vertex, float> olive;
    olive.init(argv[1], 2);
    VertexId n = olive.getVertexCount();

    // The final result, which will be aggregated.
    ranks_g = new float[n];

    // Initialize the vertex rank value to 1/n
    olive.vertexMap<PR_init_F>(PR_init_F(1.0 / n));
    olive.vertexTransform<PR_at_F>(PR_at_F());
    for (int i = 0; i < olive.getVertexCount(); i++) {
        printf("%f ", ranks_g[i]);
    }

    int iterations = 0;
    while (iterations <= 20) {
        printf("\n\n\niterations: %d worksize: %d\n",
               iterations++,
               olive.getWorksetSize());
        
        olive.edgeMap<PR_edge_F>(PR_edge_F());
        olive.vertexMap<PR_vertex_F>(PR_vertex_F());

        olive.vertexTransform<PR_at_F>(PR_at_F());
        for (int i = 0; i < olive.getVertexCount(); i++) {
            printf("%f ", ranks_g[i]);
        }
    }


    return 0;
}


