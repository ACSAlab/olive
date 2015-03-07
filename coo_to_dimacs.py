#!/usr/bin/python

"""
This script is used to convert a coo formatted
file to a dimacs formatted one.
"""

import sys

infile, outfile = sys.argv[1], sys.argv[2]

edge_read = 0
current_node = -1
nodes = 0
edges = 0
with open(infile, 'r') as inf:
    with open(outfile, 'w') as outf:
        while True:
            line = inf.readline()
            if not line:
                break
            if line.startswith("#"):
                continue
            if current_node == -1:
                arr = line.split()
                nodes, edges = int(arr[0]), int(arr[1])
                outf.write(str(nodes) + " " + str(edges))
                outf.write("\n")
                current_node += 1
            else:
                arr = line.split()
                src, dst = int(arr[0]), int(arr[1])
                if (src == current_node): 
                    outf.write(' ' + str(dst))
                else:
                    if current_node + 1 == src: #new one
                        outf.write('\n ' + str(dst))
                    else:
                        outf.write('\n')
                    current_node += 1

        outf.write("\n")
        print "Convert a graph from coo to dimacs. nodes: {0}, edges: {1}".format(nodes, edges)
