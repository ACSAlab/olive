/**
 * Defines the interface for the graph data structure
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-23
 * Last Modified: 2014-10-28
 *
 */


#ifndef OLIVE_GRAPH_H
#define OLIVE_GRAPH_H


#include "olive_def.h"


/**
 * vid_t defines the number space for vertex id.
 */
typedef uint32_t vid_t;

/**
 * eid_t defines the number space for edge id. In a sparse graph, the edge
 * number is awlays much greater than the vertex number.
 */
typedef uint32_t eid_t;

/**
 * In a weighted graph, each edge stores a value to represents its 'weight'.
 * And weight_t defines its type.
 * TODO: can be abstacted by using template, because for different
 * algorithm, different precision is required
 */
typedef float weight_t;

/**
 * For each vertex, we stores a single value in it. We might use it in some 
 * algorithms. And value_t defines this value's type.
 * TODO: We should support multiple values stored in each vertex as well as 
 * a series of device function operating on them.
 */
typedef float value_t;

/**
 * We store the graph in the memory in CSR (Compressed Sparsed Row) format for
 * its efficiency. CSR storage can minimize the memory footprint but at the 
 * expense of bringing indirect memory access.
 */
class Graph {
 private:
    /**
     * vertex_list[i] stores the starting index of vertex i's adjacent list in
     * edge_list array. We can be aware of the size of its adjacent list by
     * simply calculating vertex_list[i+1] - vertex_list[i].
     */
    eid_t    * vertex_list;
    /**
     * edge_list array stores the dest vertex id of all outgoing edges.
     * e.g. From edge_list[vertex_list[i]] stores i's neibours' ids. 
     */
    vid_t    * edge_list;
    /**
     * NOTE: each item in vertex_list array stands for a logical vertex. And it
     * stores the indices to query outgoing edges. The number to represent it 
     * must be as large as the number space of the edge id. So it is typed eid_t
     * And each item in edge_list array stands for a logical edge and stores
     * the dest vertex id. So it is typed vid_t. 
     */
    weight_t * weight_list;      // Stores the weights of edges
    value_t  * value_list;       // Stores the values of vertices
    vid_t    vertices;           // Number of vertices
    eid_t    edges;              // Number of edges
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
     * @param[in] graph_file: the path to the graph we want to read
     * @param[in] weighted: the readin graph may contain the weight
     *     information, but we can choose to build an unweighted graph
     * @return SUCCESS if built, FAILURE otherwise 
     */
    error_t initialize(const char * graph_file, bool weighted);

    /**
     * Free all the allocated buffers.
     */
    void finalize(void);
};




#endif  // OLIVE_GRAPH
