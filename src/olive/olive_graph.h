/**
 * Defines the interface for the graph data structure
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-23
 * Last Modified: 2014-10-23
 *
 */


 #ifndef OLIVE_GRAPH_H
 #define OLIVE_GRAPH_H


 #include "olive_def.h"


/**
 * vid_t defines the number space for vertex id.
 * Each vertex id represents each unique vertex in the graph.
 * vid_t can be used to type the vertex id as well as the vertex number.
 */
typedef uint32_t vid_t;

/**
 * eid_t defines the number space for 'edge id'.
 * However, we do not have a unique number to represent 'edge id'.
 * An edge is represented by the relationship between two vertices.
 * So eid_t is used to type the edge number.
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
     * vertex_list[i] stores the starting index of vertex i's adjacent list. 
     * We can be aware of the size of its adjacent list by simply calculating
     * vertex_list[i+1] - vertex_list[i].
     * NOTE: The value in vertex_list is typed eid_t since it represents the 
     * offset in edge_list. 
     */
    eid_t   * vertex_list;  
    /**
     * From edge_list[vertex_list[i]] stores its neibours' ids continuously. 
     * NOTE: The value in edge_list is typed eid_t since it represents the 
     * unique vertex id. 
     */
    vid_t    * edge_list;
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
     *     # Directed|Indirected
     *     # Weighted|Unwieghted
     *     node list
     *     edge list
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




#endif // OLIVE_GRAPH
