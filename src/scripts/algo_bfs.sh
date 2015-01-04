#!/bin/bash


OUTDIR="algo_bfs"

set -e

for graph in acyclic chain complete grid maxflow star
do
    for src in 0 2 7 11
    do
        echo graph=$graph, src=$src
        ../build/bin/algo_bfs ../data/$graph.coo $src
    done
done