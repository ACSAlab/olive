/**
 * Distributed memory implementation using Olive framework.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2015-01-03
 * Last Modified: 2015-01-03
 */

#include "olive.h"

#define INF_COST 0x7fffffff


/** Per-node state. */
struct BFS_Vertex {
    int level;
};

/**
 * Function for edge filter
 */
struct BFS_F {
    __device__
    inline bool pred(BFS_Vertex v) {
        return (v.level == INF_COST);
    }

    __device__
    inline BFS_Vertex update(BFS_Vertex v) {
        v.level += 1;
        return v;
    }
};

/**
 * Functor to set up value
 */
struct BFS_init_F {
    int _level;
    BFS_init_F(int l): _level(l) {}

    __device__
    inline BFS_Vertex operator() (BFS_Vertex v) {
        v.level = _level;
        return v;
    }
};

static int *level_g;

struct BFS_at_F {
    inline void operator() (VertexId id, BFS_Vertex value) {
        level_g[id] = value.level;
    }
};


std::vector<int> bfs_olive(const char *path, PartitionId numParts, VertexId source) {
    Olive<BFS_Vertex> olive;
    olive.init(path, numParts);
    // The final result, which will be aggregated.
    level_g = (int *) malloc(sizeof(int) * olive.getVertexCount());

    // Initialize all vertices, and filter the source vertex
    olive.vertexMap<BFS_init_F>(BFS_init_F(INF_COST));
    olive.vertexFilter<BFS_init_F>(source, BFS_init_F(0));

    int iterations = 0;
    while (!olive.isTerminated()) {
        printf("\niterations: %d\n", iterations++);
        olive.edgeFilter<BFS_F>(BFS_F());
    }

    olive.vertexTransform<BFS_at_F>(BFS_at_F());

    auto levels = std::vector<int>(level_g, level_g + olive.getVertexCount());
    for (int i = 0; i < levels.size(); i++) printf("%d ", levels[i]);
    return levels;
}
