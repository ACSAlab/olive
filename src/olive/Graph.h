/**
 * Defines the interface for the graph data structure
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-23
 * Last Modified: 2014-10-28
 */


#pragma once

#include <stdint.h>
#include "def.h"


/**
 * Defines type of the number space for vertex id.
 */
typedef uint32_t VertexId;

/**
 * Defines the number space for edge id. In a sparse graph, the edge
 * number is awlays much greater than the vertex number.
 */
typedef uint32_t EdgeId;

/**
 * In a weighted graph, each edge stores a value to represent its 'weight'.
 * TODO: can be abstacted by using template, because for different
 * algorithm, different precision is required
 */
typedef float Weight;

/**
 * For each vertex, we stores a single value in it. We may use it in some 
 * algorithms. And value_t defines this value's type.
 * TODO: We should support multiple values stored in each vertex as well as 
 * a series of device function operating on them.
 */
typedef float Value;

/**
 * We store the graph in the memory in CSR (Compressed Sparsed Row) format for
 * its efficiency. CSR storage can minimize the memory footprint but at the 
 * expense of bringing indirect memory access.
 */
class Graph {
 public:
    /**
     * Stores the starting index of vertex's outgoing edges in edgeList array.
     * We can be aware of number of vertex i's outgoing edges by calculating 
     * vertexList[i+1] - vertexList[i].
     */
    EdgeId   * vertexList;

    /**
     * Stores the destination vertex ids for outgoing edges.
     * README: each item in vertexList array stands for a *logical* vertex.
     * Physically, it is an index to query its outgoing edges. 
     * And each item in edgeList array stands for a *logical* edge.
     * Physically, it is a destination vertex id. 
     */
    VertexId * edgeList;
    Weight   * weightList;       // Stores the weights of edges
    Value    * valueList;        // Stores the values of vertices
    VertexId vertices;           // Number of vertices
    EdgeId   edges;              // Number of edges
    bool     weighted;           // Whether we keep weight in edges
    bool     valued;             // Whether we keep value in vertices

    /**
     * Constructor
     */
    Graph(void) : vertices(0), edges(0), weighted(false), valued(false),
        vertexList(NULL), edgeList(NULL), weightList(NULL), valueList(NULL) {}

    /**
     * Explictly clean up the allocated buffers
     */
    virtual void finalize(void) = 0;

    /**
     * Deconstructor
     */
    ~Graph(void) {}
};


