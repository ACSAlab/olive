/**
 * Implementation of the interface for the gpu graph data structure
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-02
 * Last Modified: 2014-11-02
 */

#include "olive_graphD.h"
#include "olive_mem.h"


Error GraphD::fromGraphH(const GraphH & graphH, GraphDMem mem) {
    // NOTE: Here we do not check the completeness of the graphH.
    dmem = mem;
    vertices = graphH.vertices;
    edges = graphH.edges;
    valued = graphH.valued;
    weighted = graphH.weighted;

    // Allocates the buffers for graphD
    switch (dmem) {
    case GRAPHD_MEM_DEVICE:
        if (vertices > 0) {
            TRY(oliveMalloc(reinterpret_cast<void **> (&vertexList),
                (vertices+1) * sizeof(EdgeId), MEM_OP_DEVICE) == SUCCESS, err_alloc);
            oliveLog("vertex list is allocated in device memory");
        }
        if (edges > 0) {
            TRY(oliveMalloc(reinterpret_cast<void **> (&edgeList),
                edges * sizeof(VertexId), MEM_OP_DEVICE) == SUCCESS, err_alloc);
            oliveLog("edge list is allocated in device memory");
        }
        break;
    case GRAPHD_MEM_HOST_MAPPED:
        if (vertices > 0) {
            TRY(oliveMalloc(reinterpret_cast<void **> (&vertexListMapped),
                (vertices+1) * sizeof(EdgeId), MEM_OP_HOST_MAPPED) == SUCCESS, err_alloc);
            TRY(oliveGetDeviceP(reinterpret_cast<void **> (vertexList),
                vertexListMapped) == SUCCESS, err_alloc);
            oliveLog("edge list is allocated in host-mapped memory");
        }
        if (edges > 0) {
            TRY(oliveMalloc(reinterpret_cast<void **> (&edgeListMapped),
                edges * sizeof(VertexId), MEM_OP_HOST_MAPPED) == SUCCESS, err_alloc);
            TRY(oliveGetDeviceP(reinterpret_cast<void **> (edgeList),
                edgeListMapped) == SUCCESS, err_alloc);
            oliveLog("edge list is allocated in host-mapped memory");
        }
        break;
    default:
        oliveError("undefined GraphD memory type");
        return FAILURE;
    }
    if (weighted) {
        TRY(oliveMalloc(reinterpret_cast<void **> (&weightList),
            edges * sizeof(Weight), MEM_OP_DEVICE) == SUCCESS, err_alloc);
        oliveLog("weight list is allocated in device memory");
    }
    if (valued) {
        TRY(oliveMalloc(reinterpret_cast<void **> (&valueList),
            vertices * sizeof(Value), MEM_OP_DEVICE) == SUCCESS, err_alloc);
        oliveLog("value list is allocated in device memory");
    }

    // Copys the graphH's buffers to graphD's buffers
    if (vertices > 0) {
        TRY(oliveMemH2D(vertexList, graphH.vertexList,
            (vertices+1) * sizeof(EdgeId)) == SUCCESS, err_copy);
        oliveLog("vertexList is copyed");
    }
    if (edges > 0) {
        TRY(oliveMemH2D(edgeList, graphH.edgeList,
            edges * sizeof(VertexId)) == SUCCESS, err_copy);
        oliveLog("edgeList is copyed");
    }
    if (weighted) {
        TRY(oliveMemH2D(weightList, graphH.weightList,
            edges * sizeof(Weight)) == SUCCESS, err_copy);
        oliveLog("weightList is copyed");
    }
    if (valued) {
        TRY(oliveMemH2D(valueList, graphH.valueList,
            vertices * sizeof(Value)) == SUCCESS, err_copy);
        oliveLog("valueList is copyed");
    }
    return SUCCESS;
err_alloc:
    oliveError("allocation err");
    finalize();
    return FAILURE;
err_copy:
    oliveError("data copy err");
    finalize();
    return FAILURE;
}

void GraphD::finalize(void) {
    oliveLog("finalizing graphD...");
    switch (dmem) {
    case GRAPHD_MEM_DEVICE:
        if (vertices > 0 && vertexList) {
            oliveFree(vertexList, MEM_OP_DEVICE);
            oliveLog("vertex list is freed from device memory");
        }
        if (edges > 0 && edgeList) {
            oliveFree(edgeList, MEM_OP_DEVICE);
            oliveLog("edge list is freed from device memory");
        }
        break;
    case GRAPHD_MEM_HOST_MAPPED:
        if (vertices > 0 && vertexListMapped) {
            oliveFree(vertexListMapped, MEM_OP_HOST_MAPPED);
            oliveLog("vertex list is freed from host-mapped memory");
        }
        if (edges > 0 && edgeListMapped) {
            oliveFree(edgeListMapped, MEM_OP_HOST_MAPPED);
            oliveLog("edge list is freed from host-mapped memory");
        }
        break;
    default:
        assert(0);
    }
    if (weighted) {
        oliveFree(weightList, MEM_OP_DEVICE);
        oliveLog("weight list is freed from device memory");
    }
    if (valued) {
        oliveFree(valueList, MEM_OP_DEVICE);
        oliveLog("value list is freed from device memory");
    }
}


