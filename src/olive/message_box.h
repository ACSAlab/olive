/**
 * Message box contains the 
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-11-28
 * Last Modified: 2014-11-29
 */

#ifndef MESSAGE_BOX_H
#define MESSAGE_BOX_H

class MessageBox {
 public:
    /**
     * Stub information sending to a remote vertex.
     * @note `id` here is .
     */
    class Message {
     public:
        VertexId  id;        // Specifying the remote vertex by its local id.
        void *    message;   // Content of the message.
    };

    /**
     * Ising a double-buffering method.
     */
    GRD<Message> * firstBuffer;
    GRD<Message> * secondBuffer;

    /** Current capacity of the message box */
    size_t size;

    // Where the message box locates
    int deviceId;

    /** Constructor */
    MessageBox(): firstBuffer(NULL), secondBuffer(NULL), deviceId(0), size(0) {}

    /** Allocating space for the message box */
    void reserve(size_t maxLength, int id) {
        firstBuffer = new GRD<Message>;
        secondBuffer = new GRD<Message>;
        firstBuffer.reserve(maxLength, id);
        secondBuffer.reserve(maxLength, id);
        deviceId = id;
        size = 0;
    }

    /** Send the content of this message box to another */
    void send(MessageBox) {

    }

    /** Swap the two buffers. */
    inline void refresh(void) {
        GRD<Message> * temp = firstBuffer;
        firstBuffer = secondBuffer;
        secondBuffer = temp;
    }

    ~MessageBox(void) {
        if (firstBuffer)  delete firstBuffer;
        if (secondBuffer) delete secondBuffer;
    }
}


#endif  // MESSAGE_BOX_H