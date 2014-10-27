/**
 * Utils
 *
 * Author: Yichao Cheng (onesuperclark@gmail.com)
 * Created on: 2014-10-20
 * Last Modified: 2014-10-23
 */

#ifndef OLIVE_UTIL_H
#define OLIVE_UTIL_H


// @return the number of the GPUs
int get_num_gpus(void);

// set the number of GPUs
void set_gpu_num(int);

// Check how much memory is available currently
void check_available_memory(void);

/**
 * Checks if the string is a numeric number
 * @param[in] str: the string to check
 * @return true if the string represents a numeric number
 */
inline bool is_numeric(char * str);

// A simple timer
// TODO: add more functions to make it more powerful
class Timer {
private:
    double last_time;
public:
    // initialize the timer before we can use time_elapsed() 
    void initialize(void);

    // @return the current time 
    double get_time(void);

    // @return the elapsed time between two continous call to this function
    double elapsed_time(void);    
};


#endif // OLIVE_UTIL_H


