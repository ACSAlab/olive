/**
 * Utils
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */


#pragma once

#include "def.h"

// @return the number of the GPUs
int getGpuNum(void);

// set the number of GPUs
void setGpuNum(int num);

// Check how much memory is available currently
void checkAvailableMemory(void);

/**
 * Checks if the string is a numeric number
 * @param str the string to check
 * @return true if the string represents a numeric number
 */
bool isNumeric(char * str);

// A simple timer
// TODO(onesuper): add more functions to make it more powerful
class Timer {
 private:
    double last_time;
 public:
    // initialize the timer before we can use time_elapsed()
    void initialize(void);

    // @return the current time in seconds
    double getTime(void);

    // @return the elapsed time between two continous call to this function
    double elapsedTime(void);
};


