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


error_t GPUGraph::parse_metadata(void) {
    assert(graph_file_handle);
}


error_t GPUGraph::parse_metadata(void) {

    char line[1024];         // Stores the readin line
    char token[];            // Stores the token separated by the delimiter 
    char remain[];           // Stores the remainder of the line      
    char delims[] = " \t\n"; // Delimiters to separate token

    // Parse metadata of the graph file
    // First line should look like # Nodes: 145
    TRY(fgets(line, 1024, graph_file_handle), err_metadata);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_metadata);
    token = strsep(&remain, delims);
    TRY(strcmp(token, "Nodes:" == 0), err_metadata);
    token = strsep(&remain, delims);
    TRY(is_numeric(token), err_metadata);
    * vertices = atoi(token);
    // Second line should look like # Edges: 145
    TRY(fgets(line, 1024, graph_file_handle), err_metadata);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_metadata);
    token = strsep(&remain, delims);
    TRY(strcmp(token, "Edges:") == 0, err_metadata);
    token = strsep(&remain, delims);
    TRY(is_numeric(token), err_metadata);
    * edges = atoi(token);
    // Third line should look like # Directed
    TRY(fgets(line, 1024, graph_file_handle), err_metadata);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_metadata);
    token = strsep(&remain, delims);
    if (strcmp(token, "Directed") == 0) {
        directed = true;
    } else if (strcmp(token, "Indirected") == 0) {
        directed = false;
    } else {
        goto err_metadata;
    }
    // Fourth line should look like # Weighted
    TRY(fgets(line, 1024, graph_file_handle), err_metadata);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_metadata);
    token = strsep(&remain, delims);
    if (strcmp(token, "Weighted") == 0) {
        weighted = true;
    } else if (strcmp(token, "Unweighted") == 0) {
        weighted = false;
    } else {
        goto err_metadata;
    }
    // All four metadata has been successfully parsed
    return SUCCESS;
    // We handle the format error here
err_metadata: 
    olive_error("metadata can not be parsed: %s\n%s", token, remain);
    return FAILURE;
}


error_t GPUGraph::parse_vertex_list(void) {

}

error_t GPUGraph::parse_edge_list(void) {

}


error_t GPUGraph::initialize(const char * graph_file, bool weighted) {
    assert(graph_file);
    graph_file_handle = fopen(graph_file, 'r');
    if (graph_file_handle) {
        olive_error("cannot open graph file: %s", graph_file);
        return FAILURE;
    }

    TRY(parse_metadata() == SUCCESS, err_parse);
    TRY(parse_vertex_list() == SUCCESS, err_parse);
    TRY(parse_edge_list() == SUCCESS, err_parse);


    // Close the file and return safely
    fclose(graph_file_handle);   
    return SUCCESS;
err_parse:
    fclose(graph_file_handle);
    olive_error("fail to initialize graph: %s", graph_file);    
    return FAILURE;
}



error_t GPUGraph::finalize(void) {

}