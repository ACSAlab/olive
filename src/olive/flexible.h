/**
 * Flexible graph representation.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-13
 * Last Modified: 2014-11-13
 */

#ifndef FLEXIBLE_H
#define FLEXIBLE_H

#include <vector>
#include <algorithm>
#include <iostream>

#include "common.h"
#include "edge_tuple.h"
#include "logging.h"
#include "utils.h"


namespace flex {

/**
 * Directed edge structure for flexible graph reprenstation. 
 * Each edge contains an `id` standing for either its destination or its source,
 * and its associated attribute.
 *
 * @tparam ED the edge attribute type
 */
template<typename ED>
class Edge {
 public:
    VertexId id;
    ED       attr;

    /** Constructor */
    explicit Edge(VertexId id_, ED d) {
        id = id_;
        attr = d;
    }

    /** For sorting the outgoing edges of a certain vertex */
    friend bool operator< (Edge a, Edge b) {
        return a.id < b.id;
    }
};


/**
 * Edge structure for flexible graph reprenstation. Each vertex contains
 * an `id`, all its outgoing edges, all its ingoing edges, and an arbitrary
 * attribtue.
 *
 * @tparam VD the vertex attribute type
 * @tparam ED the edge attribute type
 */
template<typename VD, typename ED>
class Vertex {
 public:
    std::vector<Edge<ED>> outEdges;
    std::vector<Edge<ED>> inEdges;

    VertexId id;
    VD attr;

    /** Constructor */
    explicit Vertex(VertexId id_, VD d) {
        id = id_;
        attr = d;
    }

    /** Returns the outdegree of this node. */
    size_t outdegree(void) {
        return outEdges.size();
    }

    /** Returns the indegree of this node. */
    size_t indegree(void) {
        return inEdges.size();
    }

    /** Shuffles the outgoing edges. */
    void shuffleOutEdges(void) {
        std::random_shuffle(outEdges.begin(), outEdges.end());
    }

    /** Sorts the outgoing edges according to their destination. */
    void sortOutEdges(void) {
        std::stable_sort(outEdges.begin(), outEdges.end());
    }

    /** For sorting vertices in `[[Graph]].vertices` */
    friend bool operator< (Vertex a, Vertex b) {
        return a.id < b.id;
    }
};

/**
 * Flexible graph representation for in-memory quering or manipulating.
 *
 * @tparam VD the vertex attribute type
 * @tparam ED the edge attribute type
 */
template <typename VD, typename ED>
class Graph {
 public:
    std::vector<Vertex<VD, ED>> vertices;

    /** Return the total vertex number in the graph. */
    size_t nodes(void) const {
        return vertices.size();
    }

    /** Return the total edge number in the graph. */
    size_t edges(void) const {
        size_t sum = 0;
        for (auto v : vertices) {
            sum += v.outdegree();
        }
        return sum;
    }

    /** Return the average degree of the graph in floating number. */
    float averageDegree(void) {
        return static_cast<float>(edges()) / nodes();
    }

    /**
     * Turning edge tuple reprenstation to flex's edge representation
     * 
     * @note When a graph is built with this method, the vertex-wise attribute
     * is ignored (simply set as 0).
     * 
     * @param edgeTuple An edge tuple formed as (srcId, dstId , attr)
     */
    void addEdgeTuple(EdgeTuple<ED> edgeTuple) {
        Edge<ED> outEdge(edgeTuple.dstId, edgeTuple.attr);
        Edge<ED> inEdge(edgeTuple.srcId, edgeTuple.attr);
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
            Vertex<int, ED> newNode(edgeTuple.srcId, 0);
            newNode.outEdges.push_back(outEdge);
            vertices.push_back(newNode);
        }
        if (!dstExists) {
            Vertex<int, ED> newNode(edgeTuple.dstId, 0);
            newNode.inEdges.push_back(inEdge);
            vertices.push_back(newNode);
        }
    }

    /**
     * Prints the outdegree distribution on the screen.
     * e.g., the output will look:
     * {{{
     * outDegreeLog[0]: 0         5%
     * outDegreeLog[1]: (1, 2]   15%
     * outDegreeLog[2]: (2, 4]   29%
     * ...
     * }}}
     */
     void printDegreeDist(void) {
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
            int high = pow(2, i-1);
            int low  = pow(2, i-2);
            printf("outDegreeLog[%d]: (%d, %d]\t%lu\t%.2f%%\n", i, low, high,
                   outDegreeLog[i], vertexPercentage(outDegreeLog[i]));
        }
    }

    /**
     * Load the graph from an edge list file where each line contains two 
     * integers: a source id and a target id. Skips lines that begin with `#`.
     *
     * @note If a graph is loaded from  a edge list file, the type of edge or 
     * vertex associated attribute is `int`. Meanwhile, the vertex-associated
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
     * @param path the path to the graph
     * @param vertexMemoryLevel the desired memory level for the edge partitions
     * @param edgeMemoryLevel the desired memory level for the edge partitions
     * 
     */
    void fromEdgeListFile(char * path) {
        FILE * fileHandler = fopen(path, "r");
        if (fileHandler == NULL) {
            LOG(ERROR) << "Can not open graph file: " << path;
            return;
        }
        char line[1024];                // Stores the line read in
        char temp[1024];                // Shadows the line in a temp buffer
        char * token;                   // Points to the parsed token
        char * remaining;               // Points to the remaining line

        double startTime = util::currentTimeMillis();
        while (fgets(line, 1024, fileHandler) != NULL) {
            if (line[0] != '\0' && line[0] != '#') {
                std::vector<char *> tokens;
                line[strlen(line)-1] = '\0';    // Remove the ending '\n'
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
        LOG(INFO) << "It took me " << util::currentTimeMillis() - startTime
                  << "ms to load the edge list.";
    }

    /**
     * Print the graph on the screen as the outgoing edges
     */
    void printScatter(bool withAttr = false) {
        for (auto v : vertices) {
            std::cout << "[" << v.id;
            if (withAttr) std::cout << ", " + v.attr;
            std::cout << "] ";
            for (auto e : v.outEdges) {
                std::cout << " ->" << e.id;
                if (withAttr) std::cout << ", " << e.attr;
            }
            std::cout << std::endl;
        }
    }

    /**
     * Print the graph on the screen as the ingoing edges
     */
    void printGather(bool withAttr = false) {
        for (auto v : vertices) {
            std::cout << "[" << v.id;
            if (withAttr) std::cout << ", " + v.attr;
            std::cout << "] ";
            for (auto e : v.inEdges) {
                std::cout << " <-" << e.id;
                if (withAttr) std::cout << ", " << e.attr;
            }
            std::cout << std::endl;
        }
    }

    /** Shuffles the vertices. */
    void shuffle(void) {
        std::random_shuffle(vertices.begin(), vertices.end());
    }

    /** Sorts the vertices according to their id. */
    void sort(void) {
        std::stable_sort(vertices.begin(), vertices.end());
    }

    /** Shuffles the edges. */
    void shuffleEdges(void) {
        for (auto &v : vertices) {
            v.shuffleOutEdges();
        }
    }

    /** Sorts the edges. */
    void sortEdges(void) {
        for (auto &v : vertices) {
            v.sortOutEdges();
        }
    }


 private:
    /** Return the percentage of the `n` nodes in the graph */
    float vertexPercentage(size_t n) const {
        return static_cast<float>(n) * 100.0 / nodes();
    }
};


}  // namespace flex

#endif  // FLEXIBLE_GRAPH_H
