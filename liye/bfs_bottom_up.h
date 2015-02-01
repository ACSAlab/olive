
#include <vector>

#define INF 0x7fffffff

__global__
void searchForParents(
	const EdgeId *vertices,
	const Vertex *edges,
	int *levels,
	int *mask,
	int curLevel,
	bool *isAllNodesVisited_d)
{
	int tid = THREAD_INDEX;
	int inNode = tid;
	if (mask[inNode] == 1) { // visited
		return;
	}
	EdgeId first = vertices[inNode];
	EdgeId last = vertices[inNode + 1];
	for (EdgeId edge = first; edge < last; edge ++) {
		VertexId outNode = edges[edge].localId;
		if (levels[outNode] == curLevel) {
			levels[inNode] = curLevel + 1;
			mask[inNode] = 1;
			if (*isAllNodesVisited_d == true) {
				*isAllNodesVisited_d = false;
			}
		}
	}
}

/**
 * 
 */
std::vector<int> bfs_bottom_up(const Partition<int, int> &par, VertexId nodesNum, VertexId source) {

	GRD<int> levels;
	levels.reserve(nodesNum);
	levels.allTo(INF);

	GRD<int> mask;
	mask.reserve(nodesNum);
	mask.allTo(0);

	bool *isAllNodesVisited;
    isAllNodesVisited = (bool *) malloc(sizeof(bool));

    bool *isAllNodesVisited_d;
    CUDA_CHECK(cudaMalloc(&isAllNodesVisited_d, sizeof(bool)));

    auto config = util::kernelConfig(nodesNum);

    int curLevel = 0;
    levels.set(source, 0);
    mask.set(source, 1);
	while (true) {
		*isAllNodesVisited = true;
		CUDA_CHECK(H2D(isAllNodesVisited_d, isAllNodesVisited, sizeof(bool)));

		searchForParents <<< config.first, config.second >>> (
			par.vertices.elemsDevice,
			par.edges.elemsDevice,
			levels.elemsDevice,
			mask.elemsDevice,
			curLevel,
			isAllNodesVisited_d);

		CUDA_CHECK(cudaThreadSynchronize());

		CUDA_CHECK(D2H(isAllNodesVisited, isAllNodesVisited_d, sizeof(bool)));
		printf("level = %d, isAllNodesVisited = %s\n", curLevel, isAllNodesVisited ? "true" : "false");
		if (*isAllNodesVisited == true) {
			break;
		}		

		curLevel += 1;
	}
	levels.persist();
	// Return the `levels` array
    return std::vector<int>(levels.elemsHost, levels.elemsHost + nodesNum);
}