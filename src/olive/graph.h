/**
 * Defines the interface for the graph.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-02
 * Last Modified: 2014-11-04
 */

#ifndef GRAPH_H
#define GRAPH_H

#include "common.h"


/**
 * The Graph abstractly represents a graph with arbitrary attributes
 * associated with vertices and edges.  
 *
 * @tparam VD the vertex attribute type
 * @tparam ED the edge attribute type
 */
template <typename VD, typename ED>
class Graph {
 public:
    /**
     * An RDD containing the vertices and their associated attributes.
     */
    VertexGRD<VD> vertices;

    /**
     * An RDD containing the edges and their associated attributes.
     */
    EdgeGRD<ED> edges;


};

#endif  // GRAPH_H
