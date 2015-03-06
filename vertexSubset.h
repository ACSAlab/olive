/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Yichao Cheng
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


/**
 * Vertex Subset
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2015-03-04
 * Last Modified: 2015-03-04
 */


#ifndef VERTEX_SUBSET_H
#define VERTEX_SUBSET_H

class VertexSubset {
public:
    /**
     * VertexSubset has two representations, which can convert to each other.
     * The sparse representation uses a bitmap to represent the working set.
     * The dense one uses a queue.
     */
    GRD<int>        workset;
    GRD<VertexId>   workqueue;
    VertexId       *qSize;
    VertexId       *qSizeDevice;
    bool            isDense;

    /** Make empty subset */
    VertexSubset() : qSize(NULL), qSizeDevice(NULL), isDense(false) {}

    /** Make an empty vertex subset of n vertices. */
    VertexSubset(VertexId n, bool _isDense = true) {
        if (_isDense) {
            // Reserve a dense represented GRD for output.
            isDense = true;
            workqueue.reserve(n);
            qSize = (VertexId *) malloc(sizeof(VertexId));
            *qSize = 0;
            CUDA_CHECK(cudaMalloc((void **) &qSizeDevice, sizeof(VertexId)));
            CUDA_CHECK(H2D(qSizeDevice, qSize, sizeof(VertexId)));
        } else {
            isDense = false;
            workset.reserve(n);
            workset.allTo(0);
        }
    }

    /** Make a singleton vertex in range of n. */
    VertexSubset(VertexId n, VertexId v, bool _isDense = true) {
        if (_isDense) {
            isDense = true;
            workqueue.reserve(n);
            workqueue.set(0, v);  // push v
            qSize = (VertexId *) malloc(sizeof(VertexId));
            *qSize = 1;
            CUDA_CHECK(cudaMalloc((void **) &qSizeDevice, sizeof(VertexId)));
            CUDA_CHECK(H2D(qSizeDevice, qSize, sizeof(VertexId)));
        } else {
            isDense = false;
            workset.reserve(n);
            workset.set(v, 1);
        }
    }

    /**
     * Transfer all the `workqueueSize` back and sum them up.
     */
    inline VertexId size() {
        if (isDense) {
            CUDA_CHECK(D2H(qSize, qSizeDevice, sizeof(VertexId)));
            return *qSize;
        } else {
            assert(0);
        }
    }

    inline void clear() {
        if (isDense) {
            *qSize = 0;
            CUDA_CHECK(H2D(qSizeDevice, qSize, sizeof(VertexId)));
        } else {
            workset.allTo(0);
        }
    }

    inline void print() {
        if (isDense) {
            CUDA_CHECK(D2H(qSize, qSizeDevice, sizeof(VertexId)));
            workqueue.persist();
            printf("dense: ");
            for (int i = 0; i < *qSize; i++) {
                printf("%d ", workqueue[i]);
            }
            printf("\n");
        } else {
            workset.persist();
            printf("sparse: ");
            for (int i = 0; i < workset.size(); i++) {
                if (workset[i] == 1) {
                    printf("%d ", i);
                }
            }
            printf("\n");
        }
    }

    void del() {
        if (isDense) {
            workqueue.del();
            if (qSize) free(qSize);
            if (qSizeDevice) cudaFree(qSizeDevice);
        } else {
            workset.del();
        }
    }

    // ~VertexSubset() {
    //     del();
    // }
};

#endif  // VERTEX_SUBSET_H