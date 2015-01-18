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
 * Message box
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-28
 * Last Modified: 2014-11-29
 */

#ifndef MESSAGE_BOX_H
#define MESSAGE_BOX_H

#include "common.h"

/**
 * MessageBox does not utilize GRD since the buffer only lives on host.
 * MessageBox uses host pinned memory, which is accessible by all CUDA contexts.
 * Contexts communicates with each other via asynchronized peer-to-peer access.
 *
 * A double-buffering method is used to enable the overlap of computation
 * and communication. In a super step, remote partition is allowed to copy
 * message to local
 *
 * The pointer to message box can be accessed by GPU and CPU,
 * @tparam MessageValue The data type for message contained in the box.
 */
template<typename MessageValue>
class MessageBox {
public:
    MessageValue *buffer;
    MessageValue *bufferRecv;   /** Using a double-buffering method. */
    size_t        maxLength;    /** Maximum length of the buffer */
    size_t        length;       /** Current capacity of the message box. */

    /**
     * Constructor. `deviceId < 0` if there is no memory reserved
     * It is important to give a NULL value to avoid delete a effective pointer.
     */
    MessageBox(): maxLength(0), length(0), buffer(NULL), bufferRecv(NULL) {}

    /** Allocating space for the message box */
    void reserve(size_t len) {
        assert(len > 0);
        maxLength = len;
        length = 0;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&buffer),
                                  len * sizeof(MessageValue),
                                  cudaHostAllocPortable));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&bufferRecv),
                                  len * sizeof(MessageValue),
                                  cudaHostAllocPortable));
    }

    /**
     * Copies the content of a remote message box to the `bufferRecv`.
     * So that the local partition can still work on the `buffer` in computation
     * stage.
     * Uses asynchronous memory copy to hide the memory latency with computation.
     * Assumes the peer-to-peer access is already enabled.
     *
     * If receives from an empty messagebox, the length is 0.
     * @param other   The message box to copy.
     *
     * @stream stream The stream to perform this copy within.
     */
    inline void recvMsgs(const MessageBox &other, cudaStream_t stream = 0) {
        assert(other.length <= maxLength);
        // The length should also be copied asynchronously.
        CUDA_CHECK(cudaMemcpyAsync(&length,
                                   &other.length,
                                   sizeof(size_t),
                                   cudaMemcpyDefault,
                                   stream));

        CUDA_CHECK(cudaMemcpyAsync(bufferRecv,
                                   other.buffer,
                                   length * sizeof(MessageValue),
                                   cudaMemcpyDefault,
                                   stream));

    }

    /**
     * Exchanges two buffers.
     */
    inline void swapBuffers() {
        MessageValue *temp = buffer;
        buffer = bufferRecv;
        bufferRecv = temp;
    }

    inline void clear() {
        length = 0;
    }

    inline void print() {
        for (int i = 0; i < length; i++) {
            buffer[i].print();
        }
        printf("\n");
    }


    /** Deletes the buffer */
    void del() {
        if (buffer) {
            CUDA_CHECK(cudaFreeHost(buffer));
        }
        if (bufferRecv) {
            CUDA_CHECK(cudaFreeHost(bufferRecv));
        }
    }

    /** Destructor */
    ~MessageBox() {
        del();
    }
};

#endif  // MESSAGE_BOX_H
