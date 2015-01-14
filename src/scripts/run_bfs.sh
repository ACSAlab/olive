#!/bin/bash


set -e

for graph in acyclic chain complete grid maxflow star
do
    for src in 0 2 7 11
    do
        ../build/bin/bfs ../data/$graph.coo $src
    done
done