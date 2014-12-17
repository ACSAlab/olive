

#include <vector>

__global__
void queue2mask(
    const EdgeId *vertices,
    const Vertex *edges,
    int *levels,
    int *mask,
    VertexId *queue,
    int *queueSize,
    int curLevel)
{
    int tid = THREAD_INDEX;
    if (tid >= *queueSize) return;
    VertexId outNode = queue[tid];
    EdgeId first = vertices[outNode];
    EdgeId last = vertices[outNode + 1];
    for (EdgeId edge = first; edge < last; edge ++) {
        VertexId inNode = edges[edge].localId;
        if (levels[inNode] == INF) {
            levels[inNode] = curLevel + 1;
            mask[inNode] = 1;
        }
    }
}


__global__
void mask2queue(
    size_t n,
    int *mask,
    VertexId *queue,
    int *queueSize)
{
    int tid = THREAD_INDEX;
    if (tid >= n) return;
    if (mask[tid] == 1) {
        mask[tid] = 0;
        size_t offset = atomicAdd(queueSize, 1);
        queue[offset] = tid;
    }
}


std::vector<int> bfs_top_down(const Partition &par, VertexId n) {
    GRD<int> levels;
    levels.reserve(n);
    levels.allTo(INF);
    levels.set(0, 0);

    GRD<int> mask;
    mask.reserve(n);
    mask.allTo(0);

    GRD<VertexId> queue;
    queue.reserve(n);
    queue.set(0, VertexId (0));

    int *queueSize;
    queueSize = (int *) malloc(sizeof(int));
    *queueSize = 1;

    int *queueSize_d;
    CUDA_CHECK(cudaMalloc(&queueSize_d, sizeof(int)));
    CUDA_CHECK(H2D(queueSize_d, queueSize, sizeof(int)));

    int curLevel = 0;
    while (true) {

        CUDA_CHECK(D2H(queueSize, queueSize_d, sizeof(int)));
        // printf("%d queue size: %d\n", curLevel, *queueSize);
        // queue.persist();
        // for (int i = 0; i < *queueSize; i++) {
        //     printf("%d ", queue[i]);
        // }
        // printf("\n");

        // Terminates when the queue is empty
        if (*queueSize == 0) {
            break;
        }

        auto config = util::kernelConfig(*queueSize);
        queue2mask <<< config.first, config.second>>>(
            par.vertices.elemsDevice,
            par.edges.elemsDevice,
            levels.elemsDevice,
            mask.elemsDevice,
            queue.elemsDevice,
            queueSize_d,
            curLevel);

        CUDA_CHECK(cudaThreadSynchronize());

        // Clear the queue before generating it
        *queueSize = 0;
        CUDA_CHECK(H2D(queueSize_d, queueSize, sizeof(int)));

        config = util::kernelConfig(n);
        mask2queue <<< config.first, config.second>>>(
            n,
            mask.elemsDevice,
            queue.elemsDevice,
            queueSize_d);

        curLevel += 1;
    }

    levels.persist();
    return std::vector<int>(levels.elemsHost, levels.elemsHost + n);
}

