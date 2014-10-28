#!/usr/bin/python
#
# This scripts converts a graph from COO format to CSR format
# IN: graph_name.coo
# OUT: graph_name.csr
#
# Author: Yichao Cheng (onesuperclark@gmail.com)
# Created on: 2014-10-27
# Last Modified: 2014-10-27


import argparse

# The metadata is empty first, fields are got after parsing 
metadata = {}
edge_tuples = []

# For unsorted edge tuples, we need a hash table to organize the adjacent list
edge_hash_table = {}

# The following two lists are used by CSR format
# Vertex-wise and edge-wise data are stored within vertex or edge
vertex_list = []
edge_list = []

# Here the metadata format is not strict
# We will check the completeness of the metadata 
def parse_metadata_line(line):
    tokens = line.split()
    if tokens[1] == "Nodes:": 
        metadata["vertices"] = int(tokens[2])
    elif tokens[1] == "Edges:":
        metadata["edges"] = int(tokens[2])
    elif tokens[1] == "Weighted":
        metadata["weighted"] = True 
    else:
        print "undefined metadata: " +  tokens[1] + " in " + line,

def parse_edge_tuple_line(line):
    tokens = line.split()
    src = int(tokens[0])
    dest = int(tokens[1])
    if metadata.has_key("weighted") and metadata["weighted"]:
        weight = int(tokens[2])
        edge_tuples.append((src, dest, weight)) 
    else:
        edge_tuples.append((src, dest))

# Parse the file, and set up metadata and edge tuple
def parse_file(file_handler):
    while True:
        line = file_handler.readline()
        if not line: break
        if line.startswith("#"):
            parse_metadata_line(line)
        else:
            parse_edge_tuple_line(line)

def dump_csr(file_handler):
    # Dump metadata
    file_handler.write("# Nodes: {0}\n".format(metadata["vertices"]))
    file_handler.write("# Edges: {0}\n".format(metadata["edges"]))
    if metadata.has_key("weighted") and metadata["weighted"]:
        file_handler.write("# Weighted\n")
    else:
        file_handler.write("# Unweighted\n")
    # Dump vertex list
    for vertex in vertex_list:
        file_handler.write("{0}\n".format(vertex))
    # Dump edge list
    for edge in edge_list:
        if metadata.has_key("weighted") and metadata["weighted"]: 
            file_handler.write("{0} {1}\n".format(edge[0], edge[1]))
        else:
            file_handler.write("{0}\n".format(edge))

 # This version assumes the edge tuple is ordered
def coo2csr_sorted():
    # Scans the edge tuple and fills up the vertex list and the edge list
    starting_offset = 0   # Mark the starting offset in the edge list
    last_src = -1
    for edge_tuple in edge_tuples:
        if metadata.has_key("weighted") and metadata["weighted"]:
            (src, dest, weight) = edge_tuple
        else:
            (src, dest) = edge_tuple
        # The vertex list is appended only if a different src is discoverd
        # The same starting offset will be appended repeatedly if a mucher 
        # larger is is dicoverd. For example, if the last_src is 5, and a src 
        # of 8 is discoverd, vertex_list[6...8] will all be assigned.
        # It guarantees that no vertex is missed form the vertex list
        # Since the edge tuple representation will ignore them
        for i in range(src - last_src):
            vertex_list.append(starting_offset)
        # Write dest to edge_list to represent the edge
        # The weight is associated with the edge
        if metadata.has_key("weighted") and metadata["weighted"]: 
            edge_list.append((dest, weight))
        else:
            edge_list.append(dest)

        # Cursor updation
        starting_offset += 1
        last_src = src
    # Fills up the trailing empty edges
    for i in range(metadata["vertices"] - len(vertex_list) + 1):
        vertex_list.append(starting_offset)

# This version does not assume the edge tuple is ordered
def coo2csr_unsorted():
    # Turn the edge_tuple to hash table
    for edge_tuple in edge_tuples:
        if metadata.has_key("weighted") and metadata["weighted"]:
            (src, dest, weight) = edge_tuple
        else:
            (src, dest) = edge_tuple

        if edge_hash_table.has_key(src):
            edge_hash_table[src].append(dest)
        else:
            edge_hash_table[src] = [dest]
    # Sort the adjacent list
    for k in edge_hash_table:
        edge_hash_table[k].sort()

def print_edge_hash_table():
    for k in edge_hash_table:
        print edge_hash_table[k]


if __name__ == '__main__':
    # Option registeration
    parser = argparse.ArgumentParser(
        description="This tool converts a graph from COO format to CSR format")
    parser.add_argument("file", type=str, help="file to convert")
    parser.add_argument("--out", "-o", type=str, default="a.csr", help="write output to <file>")
    #parser.add_argument("--format", "-f", type=str, choices=['csr'], 
    #    default="csr", help="format of the imput graph")
    args = parser.parse_args()

    try:
        # text => edge tuples, metadata
        parse_file(open(args.file, 'r'))
    
        # Checks the completeness of the metadata 
        assert(metadata.has_key("vertices"))
        assert(metadata.has_key("edges"))
        # print metadata

        # Check the number of edge tuples
        assert(metadata["edges"] == len(edge_tuples))

        # Call the naive coo
        coo2csr_sorted()

 
        # Check the size of vertex_list
        assert(len(vertex_list) == metadata["vertices"] + 1)
        assert(len(edge_list) == metadata["edges"])

        # Dumping out the CSR file
        dump_csr(open(args.out, 'w'))

    except IOError, e:
        print e
    



