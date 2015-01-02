/**
 * Test the bfs implementation
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-28
 * Last Modified: 2014-12-18
 */

#include "engine.h"
#include "algo_common.h"
#include "bfs_top_down.h"
#include "bfs_serial.h"

#define INF_COST 0x7fffffff

/** Per-node state. */
class BfsVertexValue {
public:
    int level;
};

static int * level_g;

/**
 * Functions for vertex map and filter
 */
__device__
BfsVertexValue init_value(BfsVertexValue value) {
    value.level = INF_COST;
    return value; 
}

__device__
BfsVertexValue init_source(BfsVertexValue value) {
    value.level = 0;
    return value;
}

/**
 * Functions for edgeFilter
 */
__device__
bool bfs_cond(BfsVertexValue value) {
    return (value.level == INF_COST);
}

__device__
BfsVertexValue bfs_update(BfsVertexValue value) {
    value.level += 1;
    return value;
}

/**
 * Functions for message packing/unpacking
 */
__device__
int bfs_pack(BfsVertexValue value) {
    return value.level;
}

__device__
BfsVertexValue bfs_unpack(int level) {
    BfsVertexValue value;
    value.level = level;
    return value;
}

typedef BfsVertexValue (*BfsMapFunc)(BfsVertexValue);
typedef bool           (*BfsCondFunc)(BfsVertexValue);
typedef int            (*BfsPackFunc)(BfsVertexValue);
typedef BfsVertexValue (*BfsUnpackFunc)(int);

__device__ BfsMapFunc init_value_d    = init_value;
__device__ BfsMapFunc init_source_d   = init_source;
__device__ BfsMapFunc bfs_update_d    = bfs_update;
__device__ BfsCondFunc bfs_cond_d     = bfs_cond;
__device__ BfsPackFunc bfs_pack_d     = bfs_pack;
__device__ BfsUnpackFunc bfs_unpack_d = bfs_unpack;



void aggr_state(VertexId globalIds, BfsVertexValue state) {
    level_g[globalIds] = state.level;
}

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("wrong argument");
        return 1;
    }

    flex::Graph<int, int> graph;
    graph.fromEdgeListFile(argv[1]);
    RandomEdgeCut random;
    auto subgraphs = graph.partitionBy(random, 1);
    Partition<int, int> par;
    par.fromSubgraph(subgraphs[0]);
    auto serial_levels = bfs_serial(par, graph.nodes(), 0);


    Engine<BfsVertexValue, int> engine;
    engine.init(argv[1], 2);
    // The final result, which will be aggregated.
    level_g = (int *) malloc(sizeof(int) * engine.getVertexCount());

    BfsMapFunc init_value_h, init_source_h, bfs_update_h;
    BfsCondFunc bfs_cond_h;
    BfsPackFunc bfs_pack_h;
    BfsUnpackFunc bfs_unpack_h;

    CUDA_CHECK(cudaMemcpyFromSymbol(&init_value_h, init_value_d, sizeof(BfsMapFunc)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&init_source_h, init_source_d, sizeof(BfsMapFunc)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&bfs_update_h, bfs_update_d, sizeof(BfsMapFunc)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&bfs_cond_h, bfs_cond_d, sizeof(BfsCondFunc)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&bfs_pack_h, bfs_pack_d, sizeof(BfsPackFunc)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&bfs_unpack_h, bfs_unpack_d, sizeof(BfsUnpackFunc)));


    engine.vertexMap(init_value_h);
    engine.vertexFilter(0, init_source_h);

    engine.run(bfs_cond_h, bfs_update_h, bfs_pack_h, bfs_unpack_h);

    engine.aggregate(aggr_state);

    auto dist_levels = std::vector<int>(level_g,
                                        level_g + engine.getVertexCount());

    for (int i = 0; i < dist_levels.size(); i++) {
        printf("%d ",dist_levels[i]);
    }

    expect_equal(serial_levels, dist_levels);

    printf("Hopefully, nothing goes wrong.");
    return 0;
}
