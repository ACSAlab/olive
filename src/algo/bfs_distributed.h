/**
 * Distributed memory implementation using Olive framework.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2015-01-03
 * Last Modified: 2015-01-03
 */

#include "engine.h"

#define INF_COST 0x7fffffff

/** Per-node state. */
class BfsVertexValue {
public:
    int level;
};

static int * level_g;

/**
 * Functor to set up value
 */
struct bfs_init_value {
    int _level;

    bfs_init_value(int l): _level(l) {}

    __device__
    BfsVertexValue operator() (BfsVertexValue value) {
        value.level = _level;
        return value;
    }
};

/**
 * Function for edge filter
 */
struct bfs_edge_context {
    __device__
    bool pred(BfsVertexValue value) {
        return (value.level == INF_COST);
    }

    __device__
    BfsVertexValue update(BfsVertexValue value) {
        value.level += 1;
        return value;
    }
};

/**
 * Functions for message packing/unpacking
 */
struct bfs_message_context {
    __device__
    int pack(BfsVertexValue value) {
        return value.level;
    }

    __device__
    BfsVertexValue unpack(int level) {
        BfsVertexValue value;
        value.level = level;
        return value;
    }
};


void bfs_gather(VertexId globalIds, BfsVertexValue state) {
    level_g[globalIds] = state.level;
}


std::vector<int> bfs_distributed(const char *path, PartitionId numParts, VertexId source) {
    Engine<BfsVertexValue, int> engine;
    engine.init(path, numParts);
    // The final result, which will be aggregated.
    level_g = (int *) malloc(sizeof(int) * engine.getVertexCount());

    // Initialize all vertices
    engine.vertexMap<bfs_init_value>(bfs_init_value(INF_COST));

    // Filter the source vertex
    engine.vertexFilter<bfs_init_value>(source, bfs_init_value(0));


    int suptersteps = 0;
    while (!engine.isTerminated()) {
        LOG(INFO) << "Superstep: " << suptersteps++;
        engine.edgeFilter<bfs_edge_context, bfs_message_context>(bfs_edge_context(), bfs_message_context());
    }

    engine.gather(bfs_gather);

    auto dist_levels = std::vector<int>(level_g,
                                        level_g + engine.getVertexCount());

    for (int i = 0; i < dist_levels.size(); i++) {
        printf("%d ",dist_levels[i]);
    }

    return dist_levels;
}
