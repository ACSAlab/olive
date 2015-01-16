# Olive: A Lightweight Graph Processing Framework for Multi-GPUs

## Input Format

The input of a graph application is an edge list file where each line represents a directional edge in the graph. More specifically, each line in the file contains two integers: a source vertex id and a target vertex id, and an optional edge-associated value. Lines that begin with `#` are treated as comments. For example:

    # Comment Line
    # SourceId  TargetId  <EdgeValue>
    1    5    <0.5>
    1    2    <0.2>
    1    8    <0.9>


## Running

Olive provides a handful of graph examples (located in `/data`) for quick test.You can run the applications on them by typing:

    $./PageRank ./data/gridGraph_15 

For some applications, like BFS, a `-s` flag (followed by an integer to indicate the source vertex) is also necessary. For example:

    $./BFS ./data/maxflowGraph_100 -s 24

## Olive Abstraction

### Edge Phase

**edgeMap**:  In this phase, . A pair of functions **gather/reduce** (isomorphic to map/reduce) is used to compute and collect the information of the neighbors of a vertex. The collected values (called **accumulator**) will be cached in the destination vertices and are further used in the vertex phase. This procedure mainly exploits the edge-level parallelism in the graph.

### Vertex Phase

**vertexMap**: perform computation based on the vertex state and the cached accumulator.

**vertexFilter**: perform computation based on the vertex state and the cached accumulator and meanwhile select a subset of vertices according to a user-defined **cond** function. The selective vertices are further used in the edge pahse.

The **vertexMap** and **vertexFilter** procedure exploit the vertex-level parallelism.


## Graph Applications


## Partition Strategy


