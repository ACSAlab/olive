/**
 * Vertex Message
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-12-29
 * Last Modified: 2014-12-29
 */


/**
 * If a vertex's `partitionId` equals the partition's `partitionId`, then it
 * refers to a local vertex. Otherwise `localId` refers to a remote one.
 */
class Vertex {
public:
    PartitionId  partitionId;
    VertexId     localId;
    explicit Vertex(): partitionId(0), localId(0) {}
    explicit Vertex(PartitionId pid, VertexId id): partitionId(pid), localId(id) {}
};

/**
 * Message information sending to a remote vertex.
 * @note `id` here is .
 */
template<typename MessageValue>
class VertexMessage {
public:
    /** Specifying the remote vertex by its local id. */
    VertexId        receiverId;
    /** Content of the message. */
    MessageValue    value;

    /** reserved for the printing on CPU. */
    void print() {
        printf("%d:%lld  ", receiverId, reinterpret_cast<long long int> (value));
    }
};