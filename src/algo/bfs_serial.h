/**
 * The serial version is used to validate the correctness of the CUDA
 * implementations.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-12-17
 * Last Modified: 2014-12-18
 */

#include <vector>
#include <deque>

#define INF 0x7fffffff

/**
 * The following algorithm comes from CLRS.
 *
 * @param partition The graph partition
 * @param nodes     The number of nodes in the graph.
 * @param source    The source node to traverse from.
 * @return a vector containing the BFS level for each node.
 */
std::vector<int> bfs_serial(const Partition<int, int> &partition,
    VertexId nodes, VertexId source) {

    GRD<int> levels;
    levels.reserve(nodes);
    levels.allTo(INF);

    GRD<int> visited;
    visited.reserve(nodes);
    visited.allTo(0);

    std::deque<VertexId> current;
    current.push_back(source);
    levels.set(source, 0);
    visited.set(source, 1);

    while(!current.empty()) {
        // Dequeue
        VertexId outNode = current.front();
        current.pop_front();

        EdgeId first = partition.vertices[outNode];
        EdgeId last = partition.vertices[outNode+1];
        for (EdgeId edge = first; edge < last; edge ++) {
            VertexId inNode = partition.edges[edge].localId;
            if (visited[inNode] == 0) {
                levels[inNode] = levels[outNode] + 1; 
                current.push_back(inNode);
                visited[inNode] = 1;
            }
        }
        // visited[outNode] = 2;
    }

    return std::vector<int>(levels.elemsHost, levels.elemsHost + nodes);
}

