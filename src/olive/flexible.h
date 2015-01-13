/**
 * Flexible graph representation.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-13
 * Last Modified: 2014-11-23
 */

#ifndef FLEXIBLE_H
#define FLEXIBLE_H

#include <vector>
#include <map>
#include <utility>
#include <algorithm>
#include <iostream>

#include "common.h"
#include "edge_tuple.h"
#include "partition_strategy.h"
#include "logging.h"
#include "utils.h"


namespace flex {
/**
 * Directed edge structure for flexible graph representation.
 * Each edge contains an vertex id for either its destination or its source,
 * and its associated edge value.
 * 
 * @tparam EdgeValue the edge value type
 */
template<typename EdgeValue>
class Edge {
public:
    VertexId vertexId;
    EdgeValue value;

    /** Constructor */
    explicit Edge(VertexId id, EdgeValue v): vertexId(id), value(v) {}

    /** For sorting the outgoing edges of a certain vertex */
    friend bool operator< (Edge a, Edge b) {
        return a.vertexId < b.vertexId;
    }
};

/**
 * Vertex entry for flexible graph representation. Each vertex contains an `id`,
 * all its outgoing edges, all its ingoing edges, and an arbitrary value.
 * A remote (ghost) vertex in a subgraph has not this entry.
 *
 * @note Storing the outgoing edges for each vertex is sufficient to represent
 * the graph. `inEdges` is just for analyzing the topology of the graph.
 *
 * @tparam VertexValue the vertex value type
 * @tparam EdgeValue the edge value type
 */
template<typename VertexValue, typename EdgeValue>
class Vertex {
public:
    /**
     * Storing only outEdges is sufficient to express the topology of a graph.
     * But keeping inEdges is useful when allocating the message buffers
     * before-hand on GPUs.
     */
    std::vector< Edge<EdgeValue> > outEdges;
    std::vector< Edge<EdgeValue> > inEdges;

    /** The unique (global) id for vertex. */
    VertexId    id;

    /** Vertex-wise value. */
    VertexValue value;

    /** Constructor */
    explicit Vertex(VertexId id_, VertexValue v) : id(id_), value(v) {}

    /** Reachability to anther vertex */
    // bool isReachableTo(VertexId other) const {
    //     for (auto e : outEdges) {
    //         if (other == e.id) return true;
    //     }
    //     return false;
    // }

    /** Returns the out-degree of this node. */
    inline size_t outdegree() const {
        return outEdges.size();
    }

    /** Shuffles the outgoing edges. */
    inline void shuffleEdges() {
        std::random_shuffle(outEdges.begin(), outEdges.end());
    }

    /** Sorts the outgoing edges according to their destination. */
    inline void sortEdgesById() {
        std::stable_sort(outEdges.begin(), outEdges.end());
    }

    /** For sorting vertices in `[[Graph]].vertices` */
    friend bool operator< (Vertex a, Vertex b) {
        return a.id < b.id;
    }
};

/**
 * Flexible graph representation for in-memory querying or manipulating.
 */
template <typename VertexValue, typename EdgeValue>
class Graph {
public:
    /** All existing vertices in a graph. */
    std::vector< Vertex<VertexValue, EdgeValue> > vertices;

    /**
     * Some vertices are missing from a partitioned subgraph. Records the
     * partition id and the local id for those missing vertices.
     * 
     * The ghost vertices are stored as key-value pairs, where the key is the
     * global id and the value is a (partitionId, localId) pair.
     * 
     * It can be used to establish a routing table which ships a ghost vertex
     * to its remote partition.
     *
     * @note This information is useful when allocating the message buffers
     * before-hand on GPUs.
     *
     * @note For a remote vertex, a remote local offset is recorded associated
     * with the `partitionId`.
     */
    std::map< VertexId, std::pair<PartitionId, VertexId> > ghostVertices;

    /** Unique id of a subgraph. */
    PartitionId partitionId;

    /** Number of the partition. */
    PartitionId numParts;

    /** Constructor */
    Graph(): partitionId(0), numParts(1) {}

    /**
     * Returns the total vertex number in the graph.
     */
    inline size_t nodes() const {
        return vertices.size();
    }

    /**
     * Returns the total edge number in the graph.
     */
    size_t edges() const {
        size_t sum = 0;
        for (auto v : vertices) {
            sum += v.outdegree();
        }
        return sum;
    }

    /**
     * Returns the average degree of the graph in floating number.
     */
    float averageDegree() const {
        return static_cast<float>(edges()) / nodes();
    }

    /**
     * Check the existence of a vertex by specifying its `id` in O(1) time.
     * If we can not find it in `ghostVertices`, then it is in this partition.
     *
     * @param  id   The id to look up
     * @return True if it exists
     */
    bool hasVertex(VertexId id) const {
        auto it = ghostVertices.find(id);
        return (it == ghostVertices.end());
    }

    /**
     * Turning edge tuple representation to flex's edge representation.
     *
     * @note When a graph is built with this method, the vertex value
     * is ignored (simply set as 0).
     *
     */
    void addEdgeTuple(EdgeTuple<EdgeValue> edgeTuple) {
        Edge<EdgeValue> outEdge(edgeTuple.dstId, edgeTuple.value);
        Edge<EdgeValue> inEdge(edgeTuple.srcId, edgeTuple.value);
        bool srcExists = false;
        bool dstExists = false;
        // If the target node already exists, append the edge directly.
        for (auto &v : vertices) {
            if (srcExists && dstExists) break;
            if (v.id == edgeTuple.srcId) {
                v.outEdges.push_back(outEdge);
                srcExists = true;
            }
            if (v.id == edgeTuple.dstId) {
                v.inEdges.push_back(inEdge);
                dstExists = true;
            }
        }
        if (!srcExists) {
            Vertex<int, EdgeValue> newNode(edgeTuple.srcId, 0);
            newNode.outEdges.push_back(outEdge);
            vertices.push_back(newNode);
        }
        if (!dstExists) {
            Vertex<int, EdgeValue> newNode(edgeTuple.dstId, 0);
            newNode.inEdges.push_back(inEdge);
            vertices.push_back(newNode);
        }
    }

    /**
     * Prints the out-degree distribution in log-style on the screen.
     * e.g., the output will look:
     * {{{
     * outDegreeLog[0]: 0         5%
     * outDegreeLog[1]: (1, 2]   15%
     * outDegreeLog[2]: (2, 4]   29%
     * ...
     * }}}
     */
    void printDegreeDist() {
        size_t outDegreeLog[32];
        int slotMax = 0;
        for (int i = 0; i < 32; i++) {
            outDegreeLog[i] = 0;
        }
        for (auto v : vertices) {
            size_t outdegree = v.outdegree();
            int slot = 0;
            while (outdegree > 0) {
                outdegree /= 2;
                slot++;
            }
            if (slot > slotMax) {
                slotMax = slot;
            }
            outDegreeLog[slot]++;
        }
        printf("outDegreeLog[0]: 0\t%lu\t%.2f%%\n", outDegreeLog[0],
               vertexPercentage(outDegreeLog[0]));

        for (int i = 1; i <= slotMax; i++) {
            int high = pow(2, i - 1);
            int low  = pow(2, i - 2);
            printf("outDegreeLog[%d]: (%d, %d]\t%lu\t%.2f%%\n", i, low, high,
                   outDegreeLog[i], vertexPercentage(outDegreeLog[i]));
        }
    }

    /**
     * Load the graph from an edge list file where each line contains two
     * integers: a source id and a target id. Skips lines that begin with `#`.
     *
     * @note If a graph is loaded from  a edge list file, the type of edge or
     * vertex associated value is `int`. Meanwhile, the vertex-associated
     * data is given a meaningless value (0 by default).
     *
     * @example Loads a file in the following format:
     * {{{
     * # Comment Line
     * # Source Id  Target Id
     * 1    5
     * 1    2
     * 2    7
     * 1    8
     * }}}
     *
     * @param path The path to the graph
     */
    void fromEdgeListFile(const char *path) {
        FILE *fileHandler = fopen(path, "r");
        if (fileHandler == NULL) {
            LOG(ERROR) << "Can not open graph file: " << path;
            return;
        }
        char line[1024];                // Stores the line read in
        char temp[1024];                // Shadows the line in a temp buffer
        char *token;                    // Points to the parsed token
        char *remaining;                // Points to the remaining line
        double startTime = util::currentTimeMillis();

        while (fgets(line, 1024, fileHandler) != NULL) {
            if (line[0] != '\0' && line[0] != '#') {
                std::vector<char *> tokens;
                line[strlen(line) - 1] = '\0';  // Remove the ending '\n'
                strncpy(temp, line, 1024);
                remaining = temp;
                while ((token = strsep(&remaining, " \t")) != NULL) {
                    tokens.push_back(token);
                }
                if (tokens.size() < 2 || !util::isNumeric(tokens[0]) || 
                    !util::isNumeric(tokens[1])) {
                    LOG(WARNING) << "Invalid line: " << line;
                    continue;
                }
                VertexId srcId = static_cast<VertexId>(atol(tokens[0]));
                VertexId dstId = static_cast<VertexId>(atol(tokens[1]));
                if (tokens.size() == 2) {
                    addEdgeTuple(EdgeTuple<int>(srcId, dstId, 1));
                } else {
                    addEdgeTuple(EdgeTuple<int>(srcId, dstId, atoi(tokens[2])));
                }
            }
        }
        LOG(INFO) << "It took " << util::currentTimeMillis() - startTime
                  << "ms to load the edge list.";
    }

    /**
     * Partitioning a graph to subgraphs by a specified `partitionStrategy`.
     * Each subgraph has independent id space (instead of the global id space).
     *
     * @note We rely on the local offset to look for a vertex from another
     * partition. So it is not allowed to shuffle vertices in any subgraph.
     * It is an convenience for converting CSR representation. See `partition.cuh`.
     *
     * @param partitionStrategy Decides which partition an vertex belongs to.
     * @param numParts          Number of partitions
     * @return                  A vector of subgraphs
     */
    std::vector< Graph<VertexValue, EdgeValue> > partitionBy(
        const PartitionStrategy &strategy,
        PartitionId numParts)
    {
        assert(numParts != 0);
        double startTime = util::currentTimeMillis();

        // Sort before partition since we want to map local id to global
        sortVerticesById();
        sortEdgesById();

        auto subgraphs = std::vector< Graph<VertexValue, EdgeValue> >(numParts);
        for (PartitionId i = 0; i < numParts; i++) {
            subgraphs[i].partitionId = i;
            subgraphs[i].numParts = numParts;
        }
        for (auto v : vertices) {
            PartitionId partitionId = strategy.getPartition(v.id, numParts);
            subgraphs[partitionId].vertices.push_back(v);
            VertexId localId = subgraphs[partitionId].vertices.size() - 1;
            auto ghost = std::pair<PartitionId, VertexId>(partitionId, localId);
            for (PartitionId i = 0; i < numParts; i++) {
                if (i == partitionId) continue;
                subgraphs[i].ghostVertices[v.id] = ghost;
            }
        }
        LOG(INFO) << "It took " << util::currentTimeMillis() - startTime
                  << "ms to partition the graph.";
        return subgraphs;
    }

    /**
     * Print the graph on the screen as the outgoing edges.
     */
    void printOutEdges(bool withValue = false) const {
        for (auto v : vertices) {
            std::cout << "[" << v.id;
            if (withValue) std::cout << ", " + v.value;
            std::cout << "] ";
            for (auto e : v.outEdges) {
                std::cout << " ->" << e.vertexId;
                if (withValue) std::cout << ", " << e.value;
            }
            std::cout << std::endl;
        }
    }

    /**
     * Print the graph on the screen as the outgoing edges.
     */
    void printInEdges(bool withValue = false) const {
        for (auto v : vertices) {
            std::cout << "[" << v.id;
            if (withValue) std::cout << ", " + v.value;
            std::cout << "] ";
            for (auto e : v.inEdges) {
                std::cout << " <-" << e.vertexId;
                if (withValue) std::cout << ", " << e.value;
            }
            std::cout << std::endl;
        }
    }

    /**
     * Print the ghost vertices on the screen (for subgraphs).
     */
    void printGhostVertices() const {
        std::cout << "ghost: {";
        for (auto g : ghostVertices) {
            std::cout << g.first << ": <" << g.second.first << ", "
                      << g.second.second << ">, ";
        }
        std::cout << "}" << std::endl;
    }

    /** Shuffles the vertices. */
    inline void shuffleVertices() {
        std::random_shuffle(vertices.begin(), vertices.end());
    }

    /** Sorts the vertices by id. */
    inline void sortVerticesById() {
        std::stable_sort(vertices.begin(), vertices.end());
    }

    /** Shuffles the edges. */
    void shuffleEdges() {
        for (auto &v : vertices) {
            v.shuffleEdges();
        }
    }

    /** Sorts the edges by id. */
    void sortEdgesById() {
        for (auto &v : vertices) {
            v.sortEdgesById();
        }
    }


private:
    /** Return the percentage of the `n` nodes in the graph */
    inline float vertexPercentage(size_t n) const {
        return static_cast<float>(n) * 100.0 / nodes();
    }
};


}  // namespace flex

#endif  // FLEXIBLE_H
