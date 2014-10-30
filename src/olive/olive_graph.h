/**
 * Defines the interface for the graph data structure
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-23
 * Last Modified: 2014-10-28
 */


#ifndef OLIVE_GRAPH_H
#define OLIVE_GRAPH_H


#include "olive_def.h"


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
 * In a weighted graph, each edge stores a value to represents its 'weight'.
 * TODO: can be abstacted by using template, because for different
 * algorithm, different precision is required
 */
typedef float Weight;

/**
 * For each vertex, we stores a single value in it. We might use it in some 
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
 private:
    /**
     * Stores the starting index of vertex i's adjacent list in
     * edge_list array. We can be aware of the size of its adjacent list by
     * simply calculating vertexList[i+1] - vertexList[i].
     */
    EdgeId   * vertexList;
    /**
     * Stores the array of dest vertex ids for outgoing edges.
     * e.g. From edgeList[vertexList[i]] stores i's neibours' ids. 
     */
    VertexId * edgeList;
    /**
     * NOTE: each item in vertexList[] array stands for a logical vertex. And it
     * stores the indices to query outgoing edges. The number to represent it 
     * must be as large as the number space of the edge id. 
     * And each item in edgeList[] array stands for a logical edge and stores
     * the dest vertex id. 
     */
    Weight   * weightList;       // Stores the weights of edges
    Value    * valueList;        // Stores the values of vertices
    VertexId vertices;           // Number of vertices
    EdgeId   edges;              // Number of edges
    bool     weighted;           // Whether we keep weight in edges
    bool     directed;           // Whether the graph is directed
    bool     valued;             // Whether we keep value in vertices

 public:
    /**
     * Reads the graph from a given file and builds it in the host memory.
     *
     * IMPORTANT: The graph file must be fed in following format
     *     # Nodes: <num_of_nodes>
     *     # Edges: <num_of_edges>
     *     # Weighted | Unweighed
     *     vertex list (could contain a single value)
     *     edge list (could contain a single weight)
     *
     * @param[in] graphFile: the path to the graph we want to read
     * @return SUCCESS if built, FAILURE otherwise 
     */
    Error initialize(const char * graphFile);

    /**
     * Free all the allocated buffers.
     */
    void finalize(void);

    /**
     * Print the graph data onto the screen
     */
    void print(void);
};




#endif  // OLIVE_GRAPH
