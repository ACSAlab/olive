/**
 * Implementation of the interface for the device-resilient graph data structure
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-23
 * Last Modified: 2014-11-04
 */


#include "olive_graphH.h"
#include "olive_mem.h"
#include "olive_util.h"


Error GraphH::fromFile(const char * graphFile) {
    FILE * graphFileHandler;
    graphFileHandler = fopen(graphFile, "r");
    if (graphFileHandler == NULL) {
        oliveError("cannot open graph file: %s", graphFile);
        return FAILURE;
    }
    oliveLog("read graph file: %s", graphFile);

    // Defines Data structures for parsing token
    // Here we use strsep(char**, char*) to get one token from line[]
    // Tokens are separated by ' \t\n'
    char line[1024];                // Stores the readin line
    char * token;                   // Stores the token
    char * remain;                  // Stores the remainder of the line
    const char delims[] = " \t\n";  // Delimiters to separate tokens

    // the first three lines are expected to be:
    // {{{
    // # Nodes: 145
    // # Edges: 145
    // # Weighted
    // }}}
    //
    // TODO(onesuper): using atoi() to convert a 64-bit integer is dangerous.
    // We need some rubustic way to convert it.
    TRY(fgets(line, 1024, graphFileHandler), err_parse_metadata);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_parse_metadata);
    token = strsep(&remain, delims);
    TRY(strcmp(token, "Nodes:") == 0, err_parse_metadata);
    token = strsep(&remain, delims);
    TRY(isNumeric(token), err_numeric);
    vertices = static_cast<VertexId> (atoi(token));
    oliveLog("input graph has %d nodes", vertices);

    TRY(fgets(line, 1024, graphFileHandler), err_parse_metadata);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_parse_metadata);
    token = strsep(&remain, delims);
    TRY(strcmp(token, "Edges:") == 0, err_parse_metadata);
    token = strsep(&remain, delims);
    TRY(isNumeric(token), err_numeric);
    edges = static_cast<EdgeId> (atoi(token));
    oliveLog("input graph has %d edges", edges);

    TRY(fgets(line, 1024, graphFileHandler), err_parse_metadata);
    remain = line;
    token = strsep(&remain, delims);
    TRY(strcmp(token, "#") == 0, err_parse_metadata);
    token = strsep(&remain, delims);
    if (strcmp(token, "Weighted") == 0) {
        weighted = true;
    } else if (strcmp(token, "Unweighted") == 0) {
        weighted = false;
    } else {
        goto err_parse_metadata;
    }
    oliveLog("input graph is %sweighted", weighted ? "" : "un");

    // FIXME(onesuper): support vertex-wise value later
    valued = false;
    oliveLog("input graph is %svalued", valued ? "" : "in");

    // Allocate the graph in the host memory.
    // NOTE: the length of vertexList is N+1. The last vertex act as a sentinel.
    // And it is possible to have 0 edges in a single-node graph.
    if (vertices > 0) {
        TRY(oliveMalloc(reinterpret_cast<void **> (&vertexList),
            (vertices+1) * sizeof(EdgeId), MEM_OP_HOST) == SUCCESS, err_alloc);
        oliveLog("vertex list is allocated (host)");
    }
    if (edges > 0) {
        TRY(oliveMalloc(reinterpret_cast<void **> (&edgeList),
            edges * sizeof(VertexId), MEM_OP_HOST) == SUCCESS, err_alloc);
        oliveLog("edge list is allocated (host)");
    }
    if (weighted) {
        TRY(oliveMalloc(reinterpret_cast<void **> (&weightList),
            edges * sizeof(Weight), MEM_OP_HOST) == SUCCESS, err_alloc);
        oliveLog("weight list is allocated (host)");
    }
    if (valued) {
        TRY(oliveMalloc(reinterpret_cast<void **> (&valueList),
            vertices * sizeof(Value), MEM_OP_HOST) == SUCCESS, err_alloc);
        oliveLog("value is allocated (host)");
    }

    // Parse the vertex list line-by-line and fill up the buffers.
    for (VertexId vid = 0; vid < vertices+1; vid++) {
        TRY(fgets(line, 1024, graphFileHandler), err_parse_list);
        remain = line;
        token = strsep(&remain, delims);
        TRY(isNumeric(token), err_numeric);
        vertexList[vid] = static_cast<EdgeId>(atoi(token));
    }
    oliveLog("all vertices are parsed from the file");

    // Parse the edge list line-by-line and fill up the buffers.
    for (EdgeId eid = 0; eid < edges; eid++) {
        TRY(fgets(line, 1024, graphFileHandler), err_parse_list);
        remain = line;
        token = strsep(&remain, delims);
        TRY(isNumeric(token), err_numeric);
        edgeList[eid] = static_cast<VertexId>(atoi(token));
        // The weight is associated with each edge line.
        if (weighted) {
            token = strsep(&remain, delims);

            // TODO(onesuper): the weight might possibly be a float number.
            // Make sure isNumeric() accepts it!
            TRY(isNumeric(token), err_numeric);
            weightList[eid] = static_cast<Weight>(atof(token));
        }
    }
    oliveLog("all edges are parsed from the file");
    fclose(graphFileHandler);
    oliveLog("close graph file");
    return SUCCESS;

    // Exception handlers
err_parse_metadata:
    oliveError("parsing metadata error: %s\n%s", token, line);
    fclose(graphFileHandler);
    return FAILURE;
err_alloc:
    oliveError("allocation error");
    fclose(graphFileHandler);
    finalize();
    return FAILURE;
err_parse_list:
    oliveError("parsing error: %s\n%s", token, line);
    fclose(graphFileHandler);
    finalize();
    return FAILURE;
err_numeric:
    oliveError("is expected to be numeric: %s\n%s", token, line);
    fclose(graphFileHandler);
    return FAILURE;
}

void GraphH::print(void) {
    if (vertices > 0) {
        printf("\nVertex list\n");
        for (VertexId vid = 0; vid < vertices+1; vid++) {
            printf("%d ", vertexList[vid]);
        }
        printf("\n");
    }
    if (edges > 0) {
        printf("Edge list\n");
        for (EdgeId eid = 0; eid < edges; eid++) {
            printf("%d ", edgeList[eid]);
        }
        printf("\n");
    }
    oliveLog("graphH is printed on the screen");
}

void GraphH::finalize(void) {
    oliveLog("finalizing graphH...");
    if (vertices > 0 && vertexList) {
        oliveFree(vertexList, MEM_OP_HOST);
        oliveLog("vertex list has been freed (host)");
    }
    if (edges > 0 && edgeList) {
        oliveFree(edgeList, MEM_OP_HOST);
        oliveLog("edge list has been freed (host)");
    }
    if (weighted && weightList) {
        oliveFree(weightList, MEM_OP_HOST);
        oliveLog("weight list has been freed (host)");
    }
    if (valued && valueList) {
        oliveFree(valueList, MEM_OP_HOST);
        oliveLog("value list has been freed (host)");
    }
}
