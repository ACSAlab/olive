/**
 * Defines the interface for the graph data structure
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-23
 * Last Modified: 2014-10-23
 */


#include "olive_graph.h"
#include "olive_mem.h"
#include "olive_util.h"


error_t Graph::initialize(const char * graph_file) {
    // Opens graph file
    FILE * graph_file_handler;
    graph_file_handler = fopen(graph_file, "r");
    if (graph_file_handler == NULL) {
        olive_error("cannot open graph file: %s", graph_file);
        return FAILURE;
    }
    olive_log("reading from graph file: %s", graph_file);
    /**
     * Defines Data structures for parsing token
     * Here we use strsep(char**, char*) to get one token from line[]
     * Tokens are separated by ' \t\n'
     * TODO(onesuper): using atoi() to convert a 64-bit integer is dangerous.
     * We need some rubustic way to convert it.
     */
    char line[1024];                // Stores the readin line
    char * token;                   // Stores the token
    char * remain;                  // Stores the remainder of the line
    const char delims[] = " \t\n";  // Delimiters to separate tokens
    /**
     * Parse metadata of the graph file (four lines). It is critical to build 
     * the graph in memory. The first line should be formatted as '# Nodes: 145'
     */
    TRY(fgets(line, 1024, graph_file_handler), err_parse);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_parse);
    token = strsep(&remain, delims);
    TRY(strcmp(token, "Nodes:") == 0, err_parse);
    token = strsep(&remain, delims);
    TRY(is_numeric(token), err_numeric);
    vertices = static_cast<vid_t>(atoi(token));
    olive_log("input graph has %d nodes", vertices);
    /**
     * The second line is expected to be '# Edges: 145'
     */
    TRY(fgets(line, 1024, graph_file_handler), err_parse);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_parse);
    token = strsep(&remain, delims);
    TRY(strcmp(token, "Edges:") == 0, err_parse);
    token = strsep(&remain, delims);
    TRY(is_numeric(token), err_numeric);
    edges = static_cast<eid_t>(atoi(token));
    olive_log("input graph has %d edges", edges);
    /**
     * The third line is expected to be '# Weighted' or '# Unweighted'
     */
    TRY(fgets(line, 1024, graph_file_handler), err_parse);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_parse);
    token = strsep(&remain, delims);
    if (strcmp(token, "Weighted") == 0) {
        weighted = true;
    } else if (strcmp(token, "Unweighted") == 0) {
        weighted = false;
    } else {
        goto err_parse;
    }
    olive_log("input graph is %sweighted", weighted ? "" : "un");
    /** 
     * TODO(onesuper): support vertex-wise value later
     */
    valued = false;
    olive_log("input graph is %svalued", valued ? "" : "in");
    /**
     * Allocate the graph in the host memory
     * The graph size is decided by the metadata  
     * NOTE: the len of vertex_list[] is N+1. 
     * And it is possible to have 0 edges in a single-node graph.
     */
    if (vertices > 0) {
        TRY(olive_malloc((void **) &vertex_list, (vertices + 1) * sizeof(eid_t),
                         OLIVE_MEM_HOST), err_host_alloc);
        olive_log("vertex list has been allocated on host");
    }
    if (edges > 0) {
        TRY(olive_malloc((void **) &edge_list, edges * sizeof(vid_t),
                         OLIVE_MEM_HOST), err_host_alloc);
        olive_log("edge list has been allocated on host");
    }
    if (weighted) {
        TRY(olive_malloc((void **) &weight_list, edges * sizeof(weight_t),
                         OLIVE_MEM_HOST), err_host_alloc);
        olive_log("weight list has been allocated on host");
    }
    if (valued) {
        TRY(olive_malloc((void **) &value_list, vertices * sizeof(value_t),
                         OLIVE_MEM_HOST), err_host_alloc);
        olive_log("value list has been allocated on host");
    }
    /**
     * Parse the graph data and sets up the vertex/edge/value/weight list
     * NOTE: the graph data is stored in CSR format. And there should be N+1
     * lines in the vertex list. The last vertex act as a sentinel
     */
    olive_log("parsing vertex list from file...");
    for (vid_t vid = 0; vid < vertices+1; vid++) {
        TRY(fgets(line, 1024, graph_file_handler), err_parse);
        remain = line;
        token = strsep(&remain, delims);
        TRY(is_numeric(token), err_numeric);
        vertex_list[vid] = static_cast<eid_t>(atoi(token));
    }
    olive_log("parsing edge list from file...");
    // Parsing the edge list.
    for (eid_t eid = 0; eid < edges; eid++) {
        TRY(fgets(line, 1024, graph_file_handler), err_parse);
        remain = line;
        token = strsep(&remain, delims);
        TRY(is_numeric(token), err_numeric);
        edge_list[eid] = static_cast<vid_t>(atoi(token));
        // The weight is associated with each edge line
        if (weighted) {
            token = strsep(&remain, delims);
            /**
             * TODO(onesuper): the weight might possibly be a float number
             * make sure is_numeric support it!
             */
            TRY(is_numeric(token), err_numeric);
            weight_list[eid] = static_cast<weight_t>(atof(token));
        }
    }
    // Close the file and return SUCCESS
    fclose(graph_file_handler);
    return SUCCESS;
    // Exception handlers
err_parse:
    fclose(graph_file_handler);
    olive_error("parsing error: %s\n%s", token, line);
    return FAILURE;
err_numeric:
    fclose(graph_file_handler);
    olive_error("is expected to be numeric: %s\n%s", token, line);
    return FAILURE;
err_host_alloc:
    fclose(graph_file_handler);
    olive_error("fail to allocate the graph on host side");
    return FAILURE;
}

void Graph::finalize(void) {
    if (vertex_list) olive_free(vertex_list, OLIVE_MEM_HOST);
    if (edge_list)   olive_free(edge_list, OLIVE_MEM_HOST);
    if (weight_list) olive_free(weight_list, OLIVE_MEM_HOST);
    if (value_list)  olive_free(value_list, OLIVE_MEM_HOST);
}

void Graph::print(void) {
    // Print the vertex list
    for (vid_t vid = 0; vid < vertices+1; vid++) {
        printf("%d ", vertex_list[vid]);
    }
    // Print the edge list
    for (eid_t eid = 0; eid < edges; eid++) {
        printf("%d ", edge_list[eid]);
    }
}

