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
 * @tparam MSG The data type for message contained in the box.
 */
template<typename MSG>
class MessageBox {
 public:
    MSG *     buffer0;     /** Using a double-buffering method. */
    MSG *     buffer1;     /** Using a double-buffering method. */
    size_t    maxLength;    /** Maximum length of the buffer */
    size_t    length;       /** Current capacity of the message box. */

    /**
     * Constructor. `deviceId < 0` if there is no memory reserved
     */
    MessageBox(): maxLength(0), length(0), buffer0(NULL), buffer1(NULL) {}

    /** Allocating space for the message box */
    void reserve(size_t len) {
        assert(len > 0);
        maxLength = len;
        length = 0;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&buffer0),
                                  len * sizeof(MSG), cudaHostAllocPortable));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&buffer1),
                                  len * sizeof(MSG), cudaHostAllocPortable));
    }

    /**
     * Copies the content of a remote message box to this.
     * We use asynchronized memory copy to hide the memory latency with
     * computation.
     *
     * Assuming the peer-to-peer access is already enabled.
     *
     * @param other   The message box to copy.
     * @stream stream The stream to perform this copy within.
     */
    void copyMsgs(const MessageBox &other, cudaStream_t stream = 0) {
        assert(other.length <= maxLength);
        assert(other.length > 0);
        length = other.length;
        CUDA_CHECK(cudaMemcpyAsync(buffer0,
                                   other.buffer0,
                                   other.length * sizeof(MSG),
                                   cudaMemcpyDefault,
                                   stream));
    }

    /**
     * Exchanges two buffers.
     */
    inline void exchange() {
        if (maxLength > 0) {
            MSG * temp = buffer0;
            buffer0 = buffer1;
            buffer1 = temp;
        }
    }

    /** Deletes the buffer */
    void del() {
        if (buffer0) {
            CUDA_CHECK(cudaFreeHost(buffer0));
        }
        if (buffer1) {
            CUDA_CHECK(cudaFreeHost(buffer1));
        }
    }

    /** Destructor */
    ~MessageBox() {
        del();
    }
};

#endif  // MESSAGE_BOX_H
