/**
 * Unit test for the partition
 * 
 *
 * Created by: onesuper (onesuperclark@gmail.com)
 * Created on: 2014-11-23
 * Last Modified: 2014-11-23
 */

#include "messageBox.h"
#include "utils.h"

__global__
void setUpMsgs(int * buffer) {
    int tid = THREAD_INDEX;
    buffer[tid] = tid * 2 + 1;
}

__global__
void verifyMsgs(int * buffer) {
    int tid = THREAD_INDEX;
    if (buffer[tid] != (tid * 2 + 1)) 
        printf("%d is not correct\n", tid);
}


/**
 * Create two on two different devices, and copy data from one to the other
 */
int main(int argc, char ** argv) {

    util::enableAllPeerAccess();
    MessageBox<int> msgbox1;
    MessageBox<int> msgbox2;

    msgbox1.reserve(4096);
    msgbox2.reserve(4096);

    auto config = util::kernelConfig(4096);

    // Set data on device 0
    CUDA_CHECK(cudaSetDevice(0));
    setUpMsgs<<<config.first, config.second>>>(msgbox1.buffer);

    msgbox1.length = 4096;
    msgbox2.recvMsgs(msgbox1);

    // Verify the data on device 1
    CUDA_CHECK(cudaSetDevice(1));
    verifyMsgs<<<config.first, config.second>>>(msgbox2.bufferRecv);

    util::disableAllPeerAccess();

    return 0;
}
