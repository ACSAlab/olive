/**
 * Defines the interface for the graph data structure
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-23
 * Last Modified: 2014-10-23
 *
 */


#include "olive_graph.h"
#include "olive_mem.h"
#include "olive_util.h"


error_t Graph::initialize(const char * graph_file, bool weighted) {
    /**
     * Opens graph file
     */
    FILE * graph_file_handler;
    graph_file_handler = fopen(graph_file, 'r');
    TRY(graph_file_handler != NULL, err_file);
    /**
     * Defines Data structures for parsing token
     * Here we use strsep() to get one token from line[]
     * Tokens are separated by ' \t\n'
     */
    char line[1024];                // Stores the readin line
    char token[];                   // Stores the token 
    const char delims[] = " \t\n";  // Delimiters to separate tokens
    /**
     * Parse metadata of the graph file (four lines). It is critical to build 
     * the graph in memory. The first line should be formatted as '# Nodes: 145'
     */
    TRY(fgets(line, 1024, graph_file_handler), err_metadata); 
    token = strsep(&line, delims);
    TRY(strcmp(token, "#") == 0, err_metadata);
    token = strsep(&line, delims);
    TRY(strcmp(token, "Nodes:" == 0), err_metadata);
    token = strsep(&line, delims);
    TRY(is_numeric(token), err_metadata);
    * vertices = atoi(token);
    
    //Second line should look like '# Edges: 145'
    TRY(fgets(line, 1024, graph_file_handler), err_metadata);
    token = strsep(&line, delims);
    TRY(strcmp(token, "#") == 0, err_metadata);
    token = strsep(&line, delims);
    TRY(strcmp(token, "Edges:") == 0, err_metadata);
    token = strsep(&line, delims);
    TRY(is_numeric(token), err_metadata);
    * edges = atoi(token);
    
    // Third line should look like '# Directed' or '# Indirected'
    TRY(fgets(line, 1024, graph_file_handler), err_metadata);
    token = strsep(&line, delims);
    TRY(strcmp(token, "#") == 0, err_metadata);
    token = strsep(&line, delims);
    if (strcmp(token, "Directed") == 0) {
        directed = true;
    } else if (strcmp(token, "Indirected") == 0) {
        directed = false;
    } else {
        goto err_metadata;
    }
    
    // Fourth line should look like '# Weighted' or '# Unweighted'
    TRY(fgets(line, 1024, graph_file_handler), err_metadata);
    token = strsep(&line, delims);
    TRY(strcmp(token, "#") == 0, err_metadata);
    token = strsep(&line, delims);
    if (strcmp(token, "Weighted") == 0) {
        weighted = true;
    } else if (strcmp(token, "Unweighted") == 0) {
        weighted = false;
    } else {
        goto err_metadata;
    }
    /**
     * Allocate the graph in the host memory
     * The graph size is decided by the metadata  
     * NOTE: the len of vertex_list[] is N+1. 
     * And it is possible to have 0 edges in a single-node graph.
     */
    if (vertices > 0)
        TRY(olive_malloc(vertex_list, (vertices + 1) * sizeof(eid_t), OLIVE_MEM_HOST),
            err_host_alloc);
    if (edges > 0)
        TRY(olive_malloc(edge_list, edges * sizeof(vid_t), OLIVE_MEM_HOST),
            err_host_alloc);
    if (weighted)
        TRY(olive_malloc(weight_list, edges * sizeof(weight_t), OLIVE_MEM_HOST), 
            err_host_alloc);
    if (valued)
        TRY(olive_malloc(value_list, vertices * sizeof(value_t), OLIVE_MEM_HOST), 
            err_host_alloc);
    /**
     * Parse the graph data and sets up the vertex/edge/value/weight list
     * NOTE: the graph data is stored in CSR format
     */


    // Close the file and return safely
    fclose(graph_file_handler);   
    return SUCCESS;
    // Exception handlers
err_file:
    olive_error("cannot open graph file: %s", graph_file);
    return FAILURE;        
err_metadata:
    fclose(graph_file_handler);
    olive_error("metadata can not be parsed: %s\n%s", token, remain);
    return FAILURE;
err_host_alloc:
    flose(graph_file_handler);
    olive_error("fail to allocate the graph on host side");
    return FAILURE;
}



void Graph::finalize(void) {
    if (vertex_list) olive_free(vertex_list, OLIVE_MEM_HOST);
    if (edge_list)   olive_free(edge_list, OLIVE_MEM_HOST);
    if (weight_list) olive_free(weight_list, OLIVE_MEM_HOST);
    if (value_list)  olive_free(value_list, OLIVE_MEM_HOST);
}


