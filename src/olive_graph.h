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
 * vid_t defines the number space for vertex_id.
 * Each vertex_id represents each unique vertex in the graph.
 * vid_t can be used to type the vertex id as well as the vertex number.
 */
typedef uint32_t vid_t;

/**
 * eid_t defines the number space for 'edge_id'.
 * However, we do not have a unique number to store an edge.
 * An edge is represented by the relationship between two vertices.
 * So eid_t is used to type the edge number.
 */
typedef uint32_t eid_t;

/**
 * In a weighted graph, each edge stores a value to represents its 'weight'.
 * weight_t defines its type.
 * TODO: can be abstacted by using template, because for different
 * algorithm, different precision is required
 */
typedef float weight_t;

/**
 * We store the graph in the memory in CSR (Compressed Sparsed Row) format for
 * its efficiency. CSR storage can minimize the memory footprint but at the 
 * expense of bringing indirect memory access.
 */
class GPUGraph {
    /**
     * vertex_list[i] stores the starting index of vertex i's adjacent list. 
     * We can be aware of the size of its adjacent list by simply calculating
     * vertex_list[i+1] - vertex_list[i].
     * NOTE: The value in vertex_list is typed eid_t since it represents the 
     * offset in edge_list. 
     */
    eid_t       * vertex_list;  
    /**
     * From edge_list[vertex_list[i]] stores its neibours' ids continuously. 
     * NOTE: The value in edge_list is typed eid_t since it represents the 
     * unique vertex id. 
     */
    vid_t       * edge_list;
    weight_t    * weight_list;   // Stores the weights of edges
    vid_t       vertices;        // Number of vertices
    eid_t       edges;           // Number of edges
    bool        weighted;        // Whether we keep weights in edges
    bool        directed;        // Whether the graph is directed
    /**
     * Reads the graph from a given file and builds it in the memory
     *
     * IMPORTANT: The graph file must be fed in following format
     *     # Nodes: <num_of_nodes>
     *     # Edges: <num_of_edges>
     *     # Directed/Indirected
     *     # Weighted/Unwieghted
     *     [src] [dest] <weight>
     *     ...
     *
     * @param[in] graph_file: the path to the graph we want to read
     * @param[in] weighted: the read-in graph may contain the weight
     *     information, but we can choose to build an unweighted graph
     * @return SUCCESS if built, FAILURE otherwise 
     */
    error_t graph_initialize(const char * graph_file, bool weighted);

    /**
     * Free all the allocated buffers.
     * @return SUCCESS if deleted, FAILURE otherwise 
     */
    error_t graph_finalize(void);
};




#endif // OLIVE_GRAPH
