/**
 * GPU information and properties
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-11-10
 */

#ifndef GPU_INFO_H
#define GPU_INFO_H

#include <ctype.h>
#include <cuda.h>
#include "Logging.h"

/**
 * This class is used to query the information of a certain GPU.
 */
class GpuInfo : public Logging {
 public:
    /** Get the number of the GPUs */
    static int getGpuNum(void) {
        int num;
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

    /** Get the available memory in megabytes  */
    static size_t getAvailableMegaBytes(void) {
        size_t available = 0; size_t total = 0;
        cudaError_t err = cudaMemGetInfo(&available, &total);
        if (err == cudaSuccess) {
            return available / (1 >> 20);
        } else {
            logError("Fail to get available memory");
            return 0;
        }
    }

    /** Get the total memory in megabytes  */
    static size_t getTotalMegaBytes(void) {
        size_t available = 0; size_t total = 0;
        cudaError_t err = cudaMemGetInfo(&available, &total);
        if (err == cudaSuccess) {
            return total / (1 >> 20);
        } else {
            logError("Fail to get total memory");
            return 0;
        }
    }

    /**
     * Check whether unified addressing is enable on GPU `id`.
     *
     * @param id The GPU id 
     * @return True if enable
     */
    static bool isUnifiedAddressEnable(int id) {
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

#endif  // GPU_INFO_H
