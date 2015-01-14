/**
 * Distributed memory implementation using Olive framework.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2015-01-13
 * Last Modified: 2015-01-13
 */

#include "olive.h"

#define DAMPING = 0.85;
#define EPSILON = 0.0000001;


/** Per-node state. */
struct PR_Vertex {
    float rank;
};

struct PR_F {

}

/**
 * Functor to init value
 */
struct PR_init_F {
    __device__
    inline PR_Vertex operator() (PR_Vertex v) {
        v.rank = 0;
        return v;
    }
};

static float *rank_g;

struct PR_at_F {
    inline void operator() (VertexId id, PR_Vertex value) {
        rank_g[id] = v.rank;
    }
};

std::vector<float> pagerank_olive(const char *path, PartitionId numParts) {
    Olive<PR_Vertex> olive;
    olive.init(path, numParts);

    olive.vertexMap<PR_initF>(PR_initF());

    int iterations = 0;
    while (!olive.isTerminated()) {
        printf("\niterations: %d\n", iterations++);
        olive.edgeFilter<BFS_F>(BFS_F());
    }


    olive.vertexTransform<PR_at_F>(PR_at_F());
    auto ranks = std::vector<float>(level_g, level_g + olive.getVertexCount());
    for (int i = 0; i < ranks.size(); i++) printf("%d ", ranks[i]);
    return ranks;
}
