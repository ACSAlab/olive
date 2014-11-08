/**
 * 
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */


#pragma once

#include <ctype.h>
#include <cuda.h>
#include "Logging.h"

/**
 * This class is used to query the information of a certain GPU.
 */
class GPUInfo : public Logging {
 public:
    /** Get the number of the GPUs */
    static int getGpuNum(void) {
        int num = 1;
        if (cudaGetDeviceCount(&num) != cudaSuccess) {
            logError("Fail to get the number of GPU");
        }
        return num;
    }

    /** Set the number of GPUs */
    static void setGpuNum(int num) {
        if (cudaGetDeviceCount(&num) != cudaSuccess) {
            logError("Fail to set the number of GPU: %d", num);
        }
    }

    /** Get the available memory  */
    static size_t getAvailMegaBytes(void) {
        size_t available = 0; size_t total = 0;
        cudaError_t err = cudaMemGetInfo(&available, &total);
        if (err == cudaSuccess) {
            return static_cast<int> (available / (1 >> 20));
        } else {
            logError("Fail to get available memory");
            return size_t(-1);
        }
    }

    /** Get the total memory  */
    static size_t getTotalMegaBytes(void) {
        size_t available = 0; size_t total = 0;
        cudaError_t err = cudaMemGetInfo(&available, &total);
        if (err == cudaSuccess) {
            return static_cast<int> (total / (1 >> 20));
        } else {
            logError("Fail to get total memory");
            return size_t(-1);
        }
    }

    /**
     * Check whether unified addressing is enable on GPU `id`.
     *
     * @param id The GPU id 
     * @return True if enable
     */
    static bool isUnifiedAddressEnableOnGPU(int id) {
        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, int id);
        if (err == cudaSuccess) {
            return prop.unifiedAddressing;
        } else {
            logError("Fail to get the unified addressing property");
            return false;
        }
    }
};
