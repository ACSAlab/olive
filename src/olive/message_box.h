/**
 * Message box contains the 
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-28
 * Last Modified: 2014-11-29
 */

#ifndef MESSAGE_BOX_H
#define MESSAGE_BOX_H



/**
 * MessageBox does not utilize GRD since the buffer only lives on GPU.
 * MessageBox uses host pinned memory, which is accessible by all CUDA contexts.
 * Contexts communicates with each other via asynchronized peer-to-peer access.
 *
 * @tparam MSG The data type for message contained in the box.
 */
template<typename MSG>
class MessageBox {
 public:
    MSG *     buffers[2];   /** Using a double-buffering method. */
    int       deviceId;     /** Where the message box locates */
    size_t    maxLength;    /** Maximum length of the buffer */
    size_t    length;       /** Current capacity of the message box. */

    /**
     * Constructor. `deviceId < 0` if there is no memory reserved
     */
    MessageBox(): deviceId(-1), maxLength(0), length(0) {
        buffers[0] = NULL;
        buffers[1] = NULL;
    }

    /** Allocating space for the message box */
    void reserve(size_t len, int id) {
        maxLength = len;
        deviceId = id;
        length = 0;
        assert(len > 0);
        assert(id >= 0);
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&buffers[0]),
                                  len * sizeof(MSG), cudaHostAllocPortable));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void **>(&buffers[1]),
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
    void copyMsgs(const MessageBox &other, cudaStream_t stream) {
        assert(deviceId != other.deviceId);
        assert(other.length <= maxLength);
        assert(other.length > 0);
        length = other.length;
        CUDA_CHECK(cudaSetDevice(deviceId));
        CUDA_CHECK(cudaMemcpyAsync(buffers[0],
                                   other.buffers[0],
                                   other.length * sizeof(MSG),
                                   cudaMemcpyDefault,
                                   stream));
    }

    /** 
     * Exchanges two buffers.
     */
    inline void exchange() {
        if (maxLength > 0) {
            MSG * temp = buffers[0];
            buffers[0] = buffers[1];
            buffers[1] = temp;
        }
    }

    /** Destructor */
    ~MessageBox() {
        if (maxLength > 0) {
            CUDA_CHECK(cudaSetDevice(deviceId));
            CUDA_CHECK(cudaFreeHost(buffers[0]));
            CUDA_CHECK(cudaFreeHost(buffers[1]));
        }
    }
};

#endif  // MESSAGE_BOX_H
