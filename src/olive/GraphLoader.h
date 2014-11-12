/**
 * Graph loader
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-05
 * Last Modified: 2014-11-05
 */

#ifndef GRAPH_LOADER_H
#define GRAPH_LOADER_H

#include <string.h>
#include <stdlib.h>
#include <vector>
#include "Logging.h"
#include "Utils.h"
#include "Graph.h"
#include "PartitionBuilder.h"


/**
 * Provides utilities for parsing graph file and loading graph.
 */
class GraphLoader: public Logging {
    /**
     * Load the graph from an edge list file where each line contains two 
     * integers: a source id and a target id. Skips lines that begin with `#`.
     * 
     * @example Loads a file in the following format:
     * {{{
     * # Comment Line
     * # Source Id  Target Id
     * 1    5
     * 1    2
     * 2    7
     * 1    8
     * }}}
     * 
     * @param path the path to the graph
     * @param vertexMemoryLevel the desired memory level for the edge partitions
     * @param edgeMemoryLevel the desired memory level for the edge partitions
     * 
     * @return A Graph object
     */
    static Graph<int, int> & fromEdgeListFile(
        char * path,
        MemoryLevel vertexMemoryLevel = CPU_ONLY,
		MemoryLevel edgeMemoryLevel = CPU_ONLY)
    {
        FILE * fileHandler = fopen(path, "r");
        if (fileHandler == NULL) {
            logError("can not open graph file: %s", path);
            return NULL;
        }
        // Structures for parsing
        char line[1024];                // Stores the line read in
        char temp[1024];                // Shadows the line in a temp buffer
        char * token;                   // Points to the parsed token
        char * remainder;               // Points to the remainder line
        const char delims[] = " \t";    // Delimiters to separate tokens
        std::vector<char *> lineVec;
        PartitionBuilder<int> builder;

        double startTime = util::currentTimeMillis();

        while (fgets(line, 1024, fileHandler) != NULL) {
            if (line[0] != '\0' && line[0] != '#') {
                lineVec.clear();
                sscanf(line, "%s", temp);
                remainder = temp;
                while ((token = strsep(&remainder, delims)) != NULL) {
                    lineVec.push_back(token);
                }
                if (lineVec.size() < 2 ||
                    !isNumeric(lineVec[0]) ||
                    !isNumeric(lineVec[1]))
                {
                    logWarning("Invalid line: %s", line);
                    continue;
                }
                /** TODO(onesuper): Currently, do not reading edge data. */
                VertexId srcId = static_cast<VertexId>(atoi(lineVec[0]));
                VertexId dstId = static_cast<VertexId>(atoi(lineVec[1]));
                builder.add(srcId, dstId, 1);
            }
        }
        logInfo("It took me %.2f ms to load the edge list", util::currentTimeMillis() - startTime);
    }
};


#endif  // GRAPH_LOADER_H
