# Olive: A Lightweight Graph Processing Framework for Multi-GPUs

![./LOGO.png]

## Input Format

The input of a graph application is an edge list file where each line represents a directional edge in the graph. More specifically, each line in the file contains two integers: a source vertex id and a target vertex id, and an optional edge-associated value. Lines that begin with `#` are treated as comments. For example:

    # Comment Line
    # SourceId  TargetId  <EdgeValue>
    1    5    <0.5>
    1    2    <0.2>
    1    8    <0.9>


## Running

Olive provides a handful of input examples (located in `/data`) for quick test.You can run the applications on them by typing:

    $./PageRank ./data/gridGraph_15 

For some applications, like BFS, a `-s` flag (followed by an integer to indicate the source vertex) is also necessary. For example:

    $./BFS ./data/maxflowGraph_100 -s 24

## Olive Abstraction


##edgeMap

**edgeMap** is used to compute and collect the information of the neighbors of a vertex. The collected value will be cached in the destination vertex (in **accumulator**) temporarily. The user can further use it in vertex phase. This procedure mainly exploits the edge-level parallelism in the graph.

**edgeMap** takes a struct `F` as input. The struct `F` contains a pair of functions `gather` and `reduce` (isomorphic to *map* and *reduce*). The `gather` function computes a value (a user defined type) for each directional edge in the graph. The `reduce` function takes the value and performs a logical sum operation on the *accumulator*. So the operator must be commutative and associative.

    struct F {
        __device__ inline V gather(PR_Vertex srcValue, EdgeId outdegree) {
            // ...
        }
        __device__ inline void reduce(float &accumulator, float accum) {
            //...
        } 
    };

### vertexMap

**vertexMap** performs computation based on the vertex state and the cached accumulator. It takes a struct `F` as input. The functor takes the accumulator as input updates the local vertex state.
    
    struct PR_vertex_F {
        __device__ inline void operator() (PR_Vertex &v, float accum) {
            //...
        }
    }


**vertexMap** has an alternative: **vertexFilter**. The only difference between them is that **vertexFilter** will filter out a subset of vertices. The filtered vertices can be further used in the edge phase.

**vertexMap** and **vertexFilter** are used to perform vertex-wise computation and mainly exploit the vertex-level parallelism.

Writing an graph application with Olive is just invoking the above three functions. The underlying runtime system deals with everything.

## Partition Strategy

The graph in Olive is edge-cut. Olive currently supports the random edge-cut partition strategy. 



