/**
 * Utils.
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */

#ifndef UTILS_H
#define UTILS_H

#include <sys/time.h>
#include <utility>

#include "cuda_runtime.h"
#include "common.h"
#include "logging.h"

namespace util {

/**
 * Calculates the block number to launch a kernel by specifying 
 * the thread number to fulfill the job and per-block thread number.
 *
 * The block number is redundant and can not exceed the limits
 * which is defined by the architecture.
 * 
 * @param  threads         How many threads is requires
 * @param  threadsPerBlock How many threads in each block (256 by default)
 * @return                 A pair (block number, thread number per block)
 */
std::pair<int, int> kernelConfig(int threads,
    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK) {
    assert(threads > 0);
    assert(threads <= MAX_THREADS);
    if (threads < threadsPerBlock) threadsPerBlock = threads;
    int blocks =  threads % threadsPerBlock == 0 ?
        threads / threadsPerBlock :
        threads / threadsPerBlock + 1;
    if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
    return std::make_pair(blocks, threadsPerBlock);
}

/**
 * Enable all-to-all peer access
 * TODO(onesuper): Temporarily making the following functions a part of utilities.
 */
void enableAllPeerAccess() {
    int numGpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGpus));
    LOG(INFO) << numGpus << " GPU detected";
    for (int i = 0; i < numGpus; i++) {
        for (int j = i+1; j < numGpus; j++) {
            CUDA_CHECK(cudaSetDevice(i));
            int canAccess = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, i, j));
            if (canAccess == 1) {
                CUDA_CHECK(cudaDeviceEnablePeerAccess(j, 0));
                LOG(INFO) << i << " enable peer access " << j;
            } else {
                LOG(WARNING) << i << " cannot access peer " << j;
            }
        }
    }
}

void disableAllPeerAccess() {
    int numGpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numGpus));
    LOG(INFO) << numGpus << " GPU detected";
    for (int i = 0; i < numGpus; i++) {
        for (int j = i+1; j < numGpus; j++) {
            CUDA_CHECK(cudaSetDevice(i));
            int canAccess = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, i, j));
            if (canAccess == 1) {
                CUDA_CHECK(cudaDeviceDisablePeerAccess(j));
                LOG(INFO) << i << " disable peer access " << j;
            } else {
                LOG(WARNING) << i << " cannot access peer " << j;
            }
        }
    }
}

/**
 * Checks if the string is a numeric number
 * @param  str String to check
 * @return     True If the string represents a numeric number
 */
bool isNumeric(const char * str) {
    assert(str);
    while ((* str) != '\0') {
        if (!isdigit(* str)) {
            return false;
        } else {
            str++;
        }
    }
    return true;
}

/**
 * Get the hash code of any given number very quickly
 * @param  a The number to hash
 * @return   The hash code
 */
size_t hashCode(size_t a) {
    a ^= (a << 13);
    a ^= (a >> 17);
    a ^= (a << 5);
    return a;
}

/** Get current time in milliseconds */
double currentTimeMillis() {
    timeval t;
    gettimeofday(&t, NULL);
    double millis = static_cast<double> (t.tv_sec * 1000.0);
    millis += static_cast<double >(t.tv_usec / 1000.0);
    return millis;
}

/** Get current time in milliseconds */
double currentTimeSeconds() {
    timeval t;
    gettimeofday(&t, NULL);
    double seconds = static_cast<double> (t.tv_sec);
    seconds += static_cast<double>(t.tv_usec / 1000000.0);
    return seconds;
}

}  // namespace util

#endif  // UTILS_H
