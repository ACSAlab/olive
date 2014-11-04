/**
 * Defines the interface for the host-resilient graph data structure
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-23
 * Last Modified: 2014-11-04
 */


#pragma once

#include "def.h"
#include "Graph.h"

class GraphH : public Graph {
 public:
    /**
     * Reads the graph from a given file and builds it in the main memory.
     *
     * @example Loads a file in the following format:
     * {{{
     * # Nodes: <num_of_nodes>
     * # Edges: <num_of_edges>
     * # Weighted | Unweighed
     * vertex list (could contain a single value)
     * edge list (could contain a single weight)
     * }}}
     *
     * @param graphFile the path to the graph we want to read
     */
    Error fromFile(const char * graphFile);

    /**
     * Print the graph data onto the screen
     */
    void print(void);

    /**
     * Explictly clean up all the allocated buffers
     */
    void finalize(void);

    /**
     * Deconstructor
     */
    ~GraphH(void) { finalize(); }
};


