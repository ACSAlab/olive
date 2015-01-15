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

struct BFS_Vertex {
    int level;
};

struct BFS_F {
    __device__
    inline bool cond(BFS_Vertex local) {
        return (local.level == INF_COST);
    }

    __device__
    inline void update(BFS_Vertex &local, BFS_Vertex srcValue) {
        local.level = srcValue.level + 1;
    }
};

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

int main(int argc, char **argv) {

    if (argc < 3) {
        printf("wrong argument");
        return 1;
    }

    Olive<BFS_Vertex> olive;
    olive.init(argv[1], 2);

    VertexId source = atoi(argv[2]);

    // The final result, which will be aggregated.
    level_g = new int[olive.getVertexCount()];

    // Initialize all vertices, and filter the source vertex
    olive.vertexMap<BFS_init_F>(BFS_init_F(INF_COST));
    olive.vertexFilter<BFS_init_F>(source, BFS_init_F(0));

    int iterations = 0;
    while (!olive.isTerminated()) {
        printf("\niterations: %d\n", iterations++);
        olive.edgeFilter<BFS_F>(BFS_F());
    }

    olive.vertexTransform<BFS_at_F>(BFS_at_F());

    for (int i = 0; i < olive.getVertexCount(); i++) {
        printf("%d ", level_g[i]);
    }

    return 0;
}
