/**
 * Message box contains the
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
 * @tparam MSG The data type for message contained in the box.
 */
template<typename MSG>
class MessageBox {
public:
    MSG      *buffer;
    MSG      *bufferRecv;   /** Using a double-buffering method. */
    size_t    maxLength;    /** Maximum length of the buffer */
    size_t    length;       /** Current capacity of the message box. */

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
                                  len * sizeof(MSG), cudaHostAllocPortable));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&bufferRecv),
                                  len * sizeof(MSG), cudaHostAllocPortable));
    }

    /**
     * Copies the content of a remote message box to the `bufferRecv`.
     * So that the local partition can still work on the `buffer` in computation
     * stage.
     * Uses asynchronous memory copy to hide the memory latency with computation.
     * Assumes the peer-to-peer access is already enabled.
     *
     * @param other   The message box to copy.
     * @stream stream The stream to perform this copy within.
     */
    void recvMsgs(const MessageBox &other, cudaStream_t stream = 0) {
        assert(other.length <= maxLength);
        if (other.length > 0) {
            length = other.length;
            CUDA_CHECK(cudaMemcpyAsync(bufferRecv,
                                       other.buffer,
                                       other.length * sizeof(MSG),
                                       cudaMemcpyDefault,
                                       stream));
        }
    }

    /**
     * Exchanges two buffers.
     */
    inline void swapBuffers() {
        MSG *temp = buffer;
        buffer = bufferRecv;
        bufferRecv = temp;
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
