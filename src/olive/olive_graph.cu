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


Error Graph::initialize(const char * graphFile) {
    // Opens graph file
    FILE * graphFileHandler;
    graphFileHandler = fopen(graphFile, "r");
    if (graphFileHandler == NULL) {
        oliveError("cannot open graph file: %s", graphFile);
        return FAILURE;
    }
    oliveLog("reading from graph file: %s", graphFile);
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
     * Parse metadata of the graph file. 
     * The first line is expected to be '# Nodes: 145'
     */
    TRY(fgets(line, 1024, graphFileHandler), err_parse);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_parse);
    token = strsep(&remain, delims);
    TRY(strcmp(token, "Nodes:") == 0, err_parse);
    token = strsep(&remain, delims);
    TRY(isNumeric(token), err_numeric);
    vertices = static_cast<VertexId>(atoi(token));
    oliveLog("input graph has %d nodes", vertices);
    /**
     * The second line is expected to be '# Edges: 145'
     */
    TRY(fgets(line, 1024, graphFileHandler), err_parse);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_parse);
    token = strsep(&remain, delims);
    TRY(strcmp(token, "Edges:") == 0, err_parse);
    token = strsep(&remain, delims);
    TRY(isNumeric(token), err_numeric);
    edges = static_cast<EdgeId>(atoi(token));
    oliveLog("input graph has %d edges", edges);
    /**
     * The third line is expected to be '# Weighted' or '# Unweighted'
     */
    TRY(fgets(line, 1024, graphFileHandler), err_parse);
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
    oliveLog("input graph is %sweighted", weighted ? "" : "un");
    /** 
     * TODO(onesuper): support vertex-wise value later
     */
    valued = false;
    oliveLog("input graph is %svalued", valued ? "" : "in");
    /**
     * Allocate the graph in the host memory.
     * The graph size is described by the metadata.
     * NOTE: the length of vertexList is N+1. 
     * And it is possible to have 0 edges in a single-node graph.
     */
    if (vertices > 0) {
        if (oliveMalloc(reinterpret_cast<void **> (&vertexList),
                        (vertices+1) * sizeof(EdgeId),
                        OLIVE_MEM_HOST) == FAILURE) {
            oliveError("fail to allocate vertex list on host side");
            fclose(graphFileHandler);
            return FAILURE;
        }
        oliveLog("vertex list is allocated on host");
    }
    if (edges > 0) {
        if (oliveMalloc(reinterpret_cast<void **> (&edgeList),
                        edges * sizeof(VertexId),
                        OLIVE_MEM_HOST) == FAILURE) {
            oliveError("fail to allocate edge list on host side");
            fclose(graphFileHandler);
            return FAILURE;
        }
        oliveLog("edge list is allocated on host");
    }
    if (weighted) {
        if (oliveMalloc(reinterpret_cast<void **> (&weightList),
                        edges * sizeof(Weight),
                        OLIVE_MEM_HOST) == FAILURE) {
            oliveError("fail to allocate weight list on host side");
            fclose(graphFileHandler);
            return FAILURE;
        }
        oliveLog("weight list is allocated on host");
    }
    if (valued) {
        if (oliveMalloc(reinterpret_cast<void **> (&valueList),
                        vertices * sizeof(Value),
                        OLIVE_MEM_HOST) == FAILURE) {
            oliveError("fail to allocate value list on host side");
            fclose(graphFileHandler);
            return FAILURE;
        }
        oliveLog("value is allocated on host");
    }
    /**
     * Parse the graph data and fill up the buffers.
     * NOTE: the graph data is stored in CSR format. And there should be N+1
     * lines in the vertex list. The last vertex act as a sentinel.
     */
    oliveLog("parsing vertex list from file...");
    for (VertexId vid = 0; vid < vertices+1; vid++) {
        TRY(fgets(line, 1024, graphFileHandler), err_parse);
        remain = line;
        token = strsep(&remain, delims);
        TRY(isNumeric(token), err_numeric);
        vertexList[vid] = static_cast<EdgeId>(atoi(token));
    }
    oliveLog("parsing edge list from file...");
    // Parsing the edge list.
    for (EdgeId eid = 0; eid < edges; eid++) {
        TRY(fgets(line, 1024, graphFileHandler), err_parse);
        remain = line;
        token = strsep(&remain, delims);
        TRY(isNumeric(token), err_numeric);
        edgeList[eid] = static_cast<VertexId>(atoi(token));
        // The weight is associated with each edge line.
        if (weighted) {
            token = strsep(&remain, delims);
            /**
             * TODO(onesuper): the weight might possibly be a float number.
             * Make sure isNumeric() accepts it!
             */
            TRY(isNumeric(token), err_numeric);
            weightList[eid] = static_cast<Weight>(atof(token));
        }
    }
    // Close the file and return SUCCESS
    fclose(graphFileHandler);
    return SUCCESS;
    // Exception handlers
err_parse:
    fclose(graphFileHandler);
    oliveError("parsing error: %s\n%s", token, line);
    return FAILURE;
err_numeric:
    fclose(graphFileHandler);
    oliveError("is expected to be numeric: %s\n%s", token, line);
    return FAILURE;
}


void Graph::finalize(void) {
    if (vertices > 0 && vertexList) oliveFree(vertexList, OLIVE_MEM_HOST);
    if (edges > 0 && edgeList) oliveFree(edgeList, OLIVE_MEM_HOST);
    if (weighted && weightList) oliveFree(weightList, OLIVE_MEM_HOST);
    if (valued && valueList) oliveFree(valueList, OLIVE_MEM_HOST);
}

void Graph::print(void) {
    printf("\nVertex list\n");
    for (VertexId vid = 0; vid < vertices+1; vid++) {
        printf("%d ", vertexList[vid]);
    }
    printf("\nEdge list\n");
    for (EdgeId eid = 0; eid < edges; eid++) {
        printf("%d ", edgeList[eid]);
    }
    printf("\n");
}

